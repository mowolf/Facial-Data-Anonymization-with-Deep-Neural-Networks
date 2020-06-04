import os
import ntpath
from pathlib import Path
import random

import cv2 as cv
import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision.utils import make_grid
from torchvision.transforms.transforms import ToTensor

from . import util
from torch.utils.tensorboard import SummaryWriter


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256, black_background=False, facenet=False,
                merge_background=False):
    """ This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.

    :param width: (int)              -- the images will be resized to width x width
    :param aspect_ratio: (float)     -- the aspect ratio of saved images
    :param image_path: (str)         -- the string is used to create image paths
    :param visuals: (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
    :param webpage: (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
    :param black_background: (bool): -- black background of image A will get added to other images
    :param merge_background:(bool):  -- merge generated image onto real background of the mask
    :param facenet: (bool):          -- show facenet distance in webpage
    """
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    webpage.add_header(name)
    ims, txts, links = [], [], []

    if black_background or merge_background:
        img = util.tensor2im(visuals['real_A'])
        data = np.array(Image.fromarray(img))
        red, green, blue = data.T
        mask = (red == 0) & (green == 0) & (blue == 0)

        if merge_background:
            # add channel to real image
            real_background = np.dstack((util.tensor2im(visuals['real_B']), np.zeros((256, 256), dtype='uint8')))
            # set background transparent
            real_background[..., 3:][mask.T] = 255
            # apply kernel to smooth the mask
            kernel = np.ones((5, 5), np.float32) / (5*5)
            # apply mask
            real_background = np.dstack((real_background[..., :3], cv.filter2D(real_background[..., 3:], -1, kernel)))
            # convert to PIL
            real_background = Image.fromarray(real_background).convert("RGBA")

        if black_background:
            background = np.zeros((256, 256, 4), dtype='uint8')
            background[..., 3:][mask.T] = 255
            background = Image.fromarray(background).convert("RGBA")
            background = background.filter(ImageFilter.MaxFilter(5))

    for label, im_data in visuals.items():

        dimensions = im_data.shape[1]
        if dimensions > 3:
            im_data = torch.cat((im_data[:, 0:3, :][0], torch.sum(im_data[:, 4:, :], dim=1)))
            im_data.unsqueeze_(0)

        im = util.tensor2im(im_data)

        if black_background:
            im = Image.fromarray(im)
            im.paste(background, (0, 0), background)
            im.convert("RGB")
            im = np.array(im)

        if merge_background and label is "fake_B":
            im = Image.fromarray(im)
            im.paste(real_background, (0, 0), real_background)
            im.convert("RGB")
            im = np.array(im)

        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)

    if facenet:
        from preprocess.facenet import facenet_distance
        facenet_result = facenet_distance(visuals['fake_B'], visuals['real_B'], percentage=True)
        txts.append(f"Facenet Distance: {facenet_result[0]:.2f} \n Facenet Confidence: {facenet_result[1]:.2f}%")
        ims.append('')
        links.append('')

    webpage.add_images(ims, txts, links, width=width)


class Visualizer:
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt, cli):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        self.saved = False


class TensorboardVisualizer(Visualizer):

    def __init__(self, opt, cli, details, dataset_len, log_dir):
        # super().__init__(opt)
        self.opt = opt
        # Writer will output to ./runs/ directory by default
        self.writer = SummaryWriter(log_dir=log_dir)

        #
        message = ''
        for k, v in sorted(vars(opt).items()):
            message += '{:>25}: {:<30}\n'.format(str(k), str(v), )
        self.writer.add_text("Info/Details", message)

        self.writer.add_text("Info/Details", 'Comment: ' + details)
        self.writer.add_text("Info/Details", 'Datapoints: ' + str(dataset_len))

        cli_text = "poetry run python " + ' '.join(cli)
        self.writer.add_text("Info/Details", cli_text)

        self.plot_dataset()

    def plot_dataset(self):
        """
        Plot Dataset to tensorboard for verification of data
        """
        # load random 30 images from dataset
        rows = 10
        cols = 5

        p = (Path(self.opt.dataroot) / self.opt.phase).glob('**/*')
        files = [x for x in p if x.is_file()]

        image_grid = torch.tensor([])
        for i in range(rows):
            image_row = torch.tensor([])
            for j in range(cols):
                idx = random.randint(0, len(files) - 1)
                image = ToTensor()(Image.open(files[idx]).resize((200, 100)))
                image_row = torch.cat((image_row, image), 1)

            image_grid = torch.cat((image_grid, image_row), 2)

        self.writer.add_image(f'Dataset/{self.opt.phase}', image_grid, 0)

    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)

    def display_current_results(self, visuals, epoch):
        width = 5
        batch_size = visuals['real_A'].size()[0]
        if batch_size < width:
            width = batch_size

        dimensions = visuals['real_A'].shape[1]
        if dimensions > 3:
            visuals['real_A'] = torch.cat(
                (visuals['real_A'][:, 0:3, :][0], torch.sum(visuals['real_A'][:, 4:, :], dim=1))).unsqueeze(0)

        # cast tensor back to image shape and discard any additional appended info
        image_grid_real_a = make_grid(visuals['real_A'][:, 0:3, :] / 2 + 0.5, nrow=width, padding=2, normalize=False,
                                      range=None,
                                      scale_each=False,
                                      pad_value=0)
        image_grid_fake_a = make_grid(visuals['fake_B'] / 2 + 0.5, nrow=width, padding=2, normalize=False, range=None,
                                      scale_each=False,
                                      pad_value=0)
        image_grid_real_b = make_grid(visuals['real_B'] / 2 + 0.5, nrow=width, padding=2, normalize=False, range=None,
                                      scale_each=False,
                                      pad_value=0)
        image_grid = torch.cat((image_grid_real_a, image_grid_fake_a, image_grid_real_b), 1)

        self.writer.add_image(f'ImageGrid/{self.opt.phase}', image_grid, epoch)

    def plot_current_losses(self, total_iters, losses):

        losses = dict(losses)
        gan_keys = ['G_GAN', 'D_fake', 'D_real']
        losses_gan = {}
        losses_g = losses
        for key in gan_keys:
            losses_gan[key] = losses[key]
            losses_g.pop(key, None)

        self.writer.add_scalars("Losses/GAN", losses_gan, total_iters)
        self.writer.add_scalars("Losses/G", losses_g, total_iters)

        # # add hparams
        # hparam_dict = {"lr": self.opt.lr, "batch_size": self.opt.batch_size, "init_type": self.opt.init_type,
        #           "Lambda_L1": self.opt.lambda_L1, "beta1": self.opt.beta1, "preprocess": self.opt.preprocess,
        #           "gan_mode": self.opt.gan_mode, "gan_mode": self.opt.gan_mode, "lr_policy": self.opt.lr_policy,
        #           "lr_decay_iters": self.opt.lr_decay_iters
        #           }
        # if "self.opt.special_mask_scale" in locals():
        #     hparam_dict.update({"mask_scale": self.opt.mask_scale, "special_mask_scale": self.opt.special_mask_scale})
        #
        # metric_dict = {"L1 loss": losses["G_L1"]}
        #
        # self.writer.add_hparams(hparam_dict=hparam_dict, metric_dict=metric_dict)

    def plot_learning_rate(self, lr, epoch):
        self.writer.add_scalar("LR", lr, epoch)
