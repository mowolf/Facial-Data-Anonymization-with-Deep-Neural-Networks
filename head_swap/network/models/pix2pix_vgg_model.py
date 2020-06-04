import torch
import torchvision.models as models
from .pix2pix_model import Pix2PixModel


class Pix2PixVggModel(Pix2PixModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        Pix2PixModel.modify_commandline_options(parser=parser, is_train=is_train)
        parser.add_argument('--vgg_resize', action='store_true', help='Resize VGG output')
        parser.add_argument('--no_l1', action='store_true', help='Dont add l1 loss')
        parser.add_argument('--vgg_weights', default=[0.25, 0.5, 1], help='weight for vgg loss for each block')
        parser.add_argument('--vgg_scaling', default=0.35, help='weight for vgg loss for each block')

        return parser

    def __init__(self, opt):
        Pix2PixModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'G_VGG']

        # Init VGG Loss model
        vgg16_features = models.vgg16(pretrained=True).features
        vgg_blocks = [vgg16_features[:9].eval().to(self.device),
                      vgg16_features[9:16].eval().to(self.device),
                      vgg16_features[16:23].eval().to(self.device)]
        for block in vgg_blocks:
            for p in block:
                p.requires_grad = False
        self.vgg_blocks = torch.nn.ModuleList(vgg_blocks)
        self.vgg_transform = torch.nn.functional.interpolate
        self.vgg_mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)).to(self.device)
        self.vgg_std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)).to(self.device)
        self.vgg_resize = opt.vgg_resize
        self.vgg_weights = opt.vgg_weights
        self.vgg_scaling = opt.vgg_scaling

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)

        # Add VGG LOSS
        input_vgg = self.fake_B
        target_vgg = self.real_B

        input_vgg = (input_vgg - self.vgg_mean) / self.vgg_std
        target_vgg = (target_vgg - self.vgg_mean) / self.vgg_std
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
        self.loss_G_VGG = self.vgg_scaling * (
                    self.vgg_weights[0] * torch.nn.functional.mse_loss(block_0_input, block_0_target) +
                    self.vgg_weights[1] * torch.nn.functional.mse_loss(block_1_input, block_1_target) +
                    self.vgg_weights[2] * torch.nn.functional.mse_loss(block_2_input, block_2_target))

        # GAN Loss
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_VGG
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        # L1 Loss
        if not self.opt.no_l1:
            self.loss_G += self.loss_G_L1

        self.loss_G.backward()
