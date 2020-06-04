import os.path
import random

import numpy as np
from skimage.metrics import structural_similarity

from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image

from preprocess.process_celebA_masks_pix2pix import create_mask_from_id
from preprocess.utils import get_files, make_color_transparent


def resample_mask(id: int):
    """
    Resamples the mask of image_id with similar face masks (face + hair is kept from original)
    :param id: id of mask image in celebaHQ
    :return: resampled mask
    """
    # get real parts we want to keep similar
    skin = create_mask_from_id(id,
                               skin=True, nose=False, glasses=False, eyes=False, brows=False, ears=True,
                               mouth=False, u_lip=False, l_lip=False, hair=False, neck=True, misc=False)

    hair = create_mask_from_id(id,
                               skin=False, nose=False, glasses=False, eyes=False, brows=False, ears=True,
                               mouth=False, u_lip=False, l_lip=False, hair=True, neck=False, misc=True)

    parts_to_resample = create_mask_from_id(id,
                                            skin=False, nose=True, glasses=True, eyes=True, brows=True, ears=False,
                                            mouth=True, u_lip=True, l_lip=True, hair=False, neck=False, misc=False)

    # get similar masks
    max_score = -1
    for i in range(40):
        # get random id
        rand_id = id
        while rand_id is id:
            # dont sample from training data?
            # 0-6000 is test that I use
            # 6001-25200 is training
            # 25201-30000 is val that I never touched
            rand_id = random.randint(0, 6000)

        random_parts = create_mask_from_id(rand_id,
                                           skin=False, nose=True, glasses=True, eyes=True, brows=True, ears=False,
                                           mouth=True, u_lip=True, l_lip=True, hair=False, neck=False, misc=False)
        # compute difference of masks
        (score, _) = structural_similarity(np.array(parts_to_resample), np.array(random_parts),
                                           full=True, multichannel=True)
        # keep highest score
        if score > max_score:
            resampled_parts = random_parts
            max_score = score
            new_id = rand_id

    print(f"Image {id} resampled to {new_id}")

    # combine real and random masks into one mask
    add_real_background = True
    if add_real_background:
        resampled_mask = Image.open('/home/mo/datasets/CelebAMask-HQ/CelebA-HQ-img/' + str(id) +".jpg").resize((256, 256), Image.NEAREST).convert("RGBA")
        skin = make_color_transparent(skin, (0, 0, 0))
    else:
        resampled_mask = Image.new("RGBA", (256, 256), "BLACK")

    resampled_mask.paste(skin, (0, 0), skin)
    resampled_parts = make_color_transparent(resampled_parts, (0, 0, 0))
    resampled_mask.paste(resampled_parts, (0, 0), resampled_parts)
    hair = make_color_transparent(hair, (0, 0, 0))
    resampled_mask.paste(hair, (0, 0), hair)

    return resampled_mask.convert("RGB")


class SwapedMaskDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert (self.opt.load_size >= self.opt.crop_size)  # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc


    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        id = int(self.AB_paths[index].split("/")[-1][:-4])

        B = resample_mask(id)

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
