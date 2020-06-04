import torch
import kornia
import numpy as np
from PIL import Image

from network.util.util import tensor2im
from .pix2pix_vgg_loss_model import Pix2PixVggLossModel


# noinspection PyAttributeOutsideInit
class Pix2PixUnalignedDomainLossKorniaModel(Pix2PixVggLossModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        Pix2PixVggLossModel.modify_commandline_options(parser=parser, is_train=is_train)
        parser.add_argument('--gauß l1 loss patch size', action='store_true', help='Resize VGG output')
        return parser

    def __init__(self, opt):
        Pix2PixVggLossModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']

    def set_input(self, input):
        Pix2PixVggLossModel.set_input(self, input)
        self.real_B_unaligned_full = input['unaligned_B'].to(self.device)
        self.alignment_params = input['alignment_params']

    def align_fake(self, margin=40, alignUnaligned=True):

        # get params
        desiredLeftEye = [float(self.alignment_params["desiredLeftEye"][0]),
                          float(self.alignment_params["desiredLeftEye"][1])]
        rotation_point = self.alignment_params["eyesCenter"]
        angle = -self.alignment_params["angle"]
        h, w = self.fake_B.shape[2:]
        # get original positions
        m1 = round(w * 0.5)
        m2 = round(desiredLeftEye[1] * w)
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
        M: torch.tensor = kornia.get_rotation_matrix2d(center, angle, scale).to(self.device)
        M[0, 0, 2] += (tX - m1)
        M[0, 1, 2] += (tY - m2)

        # get insertion point
        x_start = int(rotation_point[0] - (0.5 * w_original))
        y_start = int(rotation_point[1] - (desiredLeftEye[1] * h_original))
        # _, _, h_tensor, w_tensor = self.real_B_unaligned_full.shape

        # # # # # # # # # # # # # # # # # # ## # # # # # # # ## # # # # ## # # # # # # # # # ## # #
        # get safe margin
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
        # get face + margin unaligned space
        self.real_B_aligned_margin = self.real_B_unaligned_full[:, :,
                                     y_start - margin:y_start + h_original + margin,
                                     x_start - margin:x_start + w_original + margin]
        # invert matrix
        M_inverse = kornia.invert_affine_transform(M)
        # update output size to fit the 256 + scaled margin
        old_size = self.real_B_aligned_margin.shape[2]
        new_size = old_size + 2 * round(float(margin * scale))

        _, _, h_tensor, w_tensor = self.real_B_aligned_margin.shape
        self.real_B_aligned_margin = kornia.warp_affine(self.real_B_aligned_margin, M_inverse,
                                                        dsize=(new_size, new_size))
        # padding_mode="reflection")
        self.fake_B_aligned_margin = self.real_B_aligned_margin.clone().requires_grad_(True)

        # update margin as we now scale the image!
        # update start point
        start = round(float(margin * scale * new_size / old_size))
        print(start)

        # point = torch.tensor([0, 0, 1], dtype=torch.float)
        # M_ = M_inverse[0].clone().detach()
        # M_ = torch.cat((M_, torch.tensor([[0, 0, 1]], dtype=torch.float)))
        #
        # M_n = M[0].clone().detach()
        # M_n = torch.cat((M_n, torch.tensor([[0, 0, 1]], dtype=torch.float)))
        #
        # start_tensor = torch.matmul(torch.matmul(point, M_) + margin, M_n)
        # print(start_tensor)
        # start_y, start_x = round(float(start_tensor[0])), round(float(start_tensor[1]))

        # reinsert into tensor
        self.fake_B_aligned_margin[0, :, start:start + 256, start:start + 256] = self.real_B

        Image.fromarray(tensor2im(self.real_B_aligned_margin)).save("/home/mo/datasets/ff_aligned_unaligned/real.png")
        Image.fromarray(tensor2im(self.fake_B_aligned_margin)).save("/home/mo/datasets/ff_aligned_unaligned/fake.png")

        exit()
        # # # # # # # # # # # # # # # # # # ## # # # # # # # ## # # # # ## # # # # # # # # # ## # #
        if not alignUnaligned:
            # Now apply the transformation to original image
            # clone fake
            fake_B_clone = self.fake_B.clone().requires_grad_(True)
            # apply warp
            fake_B_warped: torch.tensor = kornia.warp_affine(fake_B_clone, M, dsize=(h_original, w_original))

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

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG(self.real_A)  # G(A)
        self.align_fake()

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B_unaligned),
                            1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B_unaligned), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # Normal backward pass
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B_unaligned), 1)
        pred_fake = self.netD(fake_AB)
        # GAN Loss
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # VGG loss
        self.loss_G_VGG = self.vgg_loss(self.fake_B_unaligned, self.real_B_unaligned)
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_VGG
        self.loss_G_L1 = self.criterionL1(self.fake_B_unaligned, self.real_B_unaligned) * self.opt.lambda_L1
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
