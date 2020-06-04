import torch.nn.functional as F
import torch
import torchgeometry as tgm
import numpy as np

from .pix2pix_vgg_loss_model import Pix2PixVggLossModel

# noinspection PyAttributeOutsideInit
class Pix2PixUnalignedDomainLossModel(Pix2PixVggLossModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        Pix2PixVggLossModel.modify_commandline_options(parser=parser, is_train=is_train)
        parser.add_argument('--only_disc', action='store_true', help='Only use unaligned for discriminator.')
        parser.add_argument('--vgg_unalign_scale', default=0.5, help='Unaligned Scale factor for vgg loss')
        parser.add_argument('--mask_scale', type=int, default=2,
                            help='Linear scaling of mask region for weighted L1 loss')

        return parser

    def __init__(self, opt):
        Pix2PixVggLossModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.only_disc = opt.only_disc
        self.vgg_unalign_scale = opt.vgg_unalign_scale
        self.mask_scale = opt.mask_scale

    def set_input(self, input):
        Pix2PixVggLossModel.set_input(self, input)
        if self.isTrain:
            self.real_B_unaligned_full = input['unaligned_B'].to(self.device)
            self.alignment_params = input['alignment_params']

    def align_fake(self, margin=20):
        # get params
        desiredLeftEye = [float(self.alignment_params["desiredLeftEye"][0]),
                          float(self.alignment_params["desiredLeftEye"][1])]
        rotation_point = self.alignment_params["eyesCenter"]
        angle = -self.alignment_params["angle"]
        h, w = self.fake_B.shape[2:]
        # get original positions
        m1 = round(w * 0.5)
        m2 = round(desiredLeftEye[0] * w)
        # define the scale factor
        scale = 1 / self.alignment_params["scale"]
        width = int(self.alignment_params["shape"][0])
        long_edge_size = width / abs(np.cos(np.deg2rad(angle)))
        w_original = int(scale * long_edge_size)
        h_original = int(scale * long_edge_size)
        # get offset
        tX = w_original * 0.5
        tY = h_original * desiredLeftEye[1]
        # get rotation center
        center = torch.ones(1, 2)
        center[..., 0] = m1
        center[..., 1] = m2
        # compute the transformation matrix
        M = tgm.get_rotation_matrix2d(center, angle, scale).to(self.device)
        M[0, 0, 2] += (tX - m1)
        M[0, 1, 2] += (tY - m2)
        # get insertion point
        x_start = int(rotation_point[0] - (0.5 * w_original))
        y_start = int(rotation_point[1] - (desiredLeftEye[0] * h_original))
        _, _, h_tensor, w_tensor = self.real_B_unaligned_full.shape

        # Now apply the transformation to original image
        # clone fake
        fake_B_clone = self.fake_B.clone().requires_grad_(True)
        # apply warp
        fake_B_warped = tgm.warp_affine(fake_B_clone, M, dsize=(h_original, w_original))
        # clone warped
        self.fake_B_unaligned = fake_B_warped.clone().requires_grad_(True)

        # make sure warping does not exceed real_B_unaligned_full dimensions
        if y_start < 0:
            fake_B_warped = fake_B_warped[:, :, abs(y_start):h_original, :]
            h_original += y_start
            y_start = 0
        if x_start < 0:
            fake_B_warped = fake_B_warped[:, :, :, abs(x_start):w_original]
            w_original += x_start
            x_start = 0
        if y_start + h_original > h_tensor:
            h_original -= (y_start + h_original - h_tensor)
            fake_B_warped = fake_B_warped[:, :, 0:h_original, :]
        if x_start + w_original > w_tensor:
            w_original -= (x_start + w_original - w_tensor)
            fake_B_warped = fake_B_warped[:, :, :, 0:w_original]

        # create mask that is true where fake_B_warped is 0
        # This is the background that is not filled with image after the transformation
        mask = ((fake_B_warped[0][0] == 0) & (fake_B_warped[0][1] == 0) & (fake_B_warped[0][2] == 0))
        # fill fake_B_filled where mask = False with self.real_B_unaligned_full
        fake_B_filled = torch.where(mask,
                                    self.real_B_unaligned_full[:, :, y_start:y_start + h_original,
                                    x_start:x_start + w_original],
                                    fake_B_warped)

        # reinsert into tensor
        self.fake_B_unaligned = self.real_B_unaligned_full.clone().requires_grad_(True)
        mask = torch.zeros_like(self.fake_B_unaligned, dtype=torch.bool)
        mask[0, :, y_start:y_start + h_original, x_start:x_start + w_original] = True
        self.fake_B_unaligned = self.fake_B_unaligned.masked_scatter(mask, fake_B_filled)

        # cutout tensor
        h_size_tensor, w_size_tensor = self.real_B_unaligned_full.shape[2:]
        margin = max(
            min(
                y_start - max(0, y_start - margin),
                x_start - max(0, x_start - margin),
                min(y_start + h_original + margin, h_size_tensor) - y_start - h_original,
                min(x_start + w_original + margin, w_size_tensor) - x_start - w_original,
            ),
            0
        )
        self.fake_B_unaligned = self.fake_B_unaligned[:, :, y_start - margin:y_start + h_original + margin,
                                x_start - margin:x_start + w_original + margin]
        self.real_B_unaligned = self.real_B_unaligned_full[:, :, y_start - margin:y_start + h_original + margin,
                                x_start - margin:x_start + w_original + margin]

        self.real_B_unaligned = F.interpolate(self.real_B_unaligned, size=(300, 300))
        self.fake_B_unaligned = F.interpolate(self.fake_B_unaligned, size=(300, 300))


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)
        if self.isTrain:
            self.align_fake()

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B),
                            1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # Normal backward pass
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        # GAN Loss
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # VGG loss
        if self.only_disc:
            self.loss_G_VGG = self.vgg_loss(self.fake_B_unaligned, self.real_B_unaligned)
        else:
            self.loss_G_VGG = self.vgg_unalign_scale * self.vgg_loss(self.fake_B_unaligned, self.real_B_unaligned) \
                              + (1 - self.vgg_unalign_scale) * self.vgg_loss(self.fake_B, self.real_B)

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_VGG
        # self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # scaled L1 loss
        mask = ((self.real_A[0][0] > -1) & (self.real_A[0][1] > -1) & (self.real_A[0][2] > -1))
        scale = mask * (torch.ones(256, 256) * self.mask_scale).to(device=self.gpu_ids[0]) + \
                torch.ones(256, 256).to(device=self.gpu_ids[0])

        self.loss_G_L1 = self.criterionL1(self.fake_B * scale, self.real_B * scale) * self.opt.lambda_L1

        # L1 Loss
        if not self.opt.no_l1:
            self.loss_G += self.loss_G_L1

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()  # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()  # calculate graidents for G
        self.optimizer_G.step()  # udpate G's weights

    def vgg_loss(self, input, target):
        # Add VGG LOSS
        input_vgg = (input - self.vgg_mean) / self.vgg_std
        target_vgg = (target - self.vgg_mean) / self.vgg_std
        if self.vgg_resize:
            input_vgg = self.vgg_transform(input_vgg, mode='bilinear', size=(224, 224), align_corners=False)
            target_vgg = self.vgg_transform(target_vgg, mode='bilinear', size=(224, 224), align_corners=False)
        # self.loss_VGG = 0.0
        # for block in self.vgg_blocks:
        block_0_input = self.vgg_blocks[0](input_vgg)
        block_0_target = self.vgg_blocks[0](target_vgg)
        block_1_input = self.vgg_blocks[1](block_0_input)
        block_1_target = self.vgg_blocks[1](block_0_target)
        block_2_input = self.vgg_blocks[2](block_1_input)
        block_2_target = self.vgg_blocks[2](block_1_target)
        # VGG and L1 should have about the same scaling
        return self.vgg_scaling * (
                self.vgg_weights[0] * torch.nn.functional.mse_loss(block_0_input, block_0_target) +
                self.vgg_weights[1] * torch.nn.functional.mse_loss(block_1_input, block_1_target) +
                self.vgg_weights[2] * torch.nn.functional.mse_loss(block_2_input, block_2_target))
