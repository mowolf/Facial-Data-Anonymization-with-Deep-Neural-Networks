import os.path
from pathlib import Path

from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image

from preprocess.process_celebA_masks_pix2pix import create_mask_from_id, im_resize


class WarpedMaskDataset(BaseDataset):
    """A dataset class for paired image dataset for test run only. This warps input masks randomly to create
        a more distinct person.

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

        # server
        self.data_path = Path('/home/mo/datasets/CelebAMask-HQ/CelebAMask-HQ-mask-anno')
        # local
        #self.data_path = Path("/Users/Mo/Google Drive/MasterThesis/code/masterthesis/face_generation/preprocess/images/CelebAMask-HQ-mask-anno")


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
        A_path = self.AB_paths[index]
        A = Image.open(A_path).convert('RGB')

        w, h = A.size
        if w is not h:
            w2 = int(w / 2)
            A = A.crop((0, 0, w2, h))

        pic_id = int(self.AB_paths[index].split("/")[-1][:-4])

        showGUI = False
        warp = True
        if self.opt.gui:
            showGUI = True
            warp = False

        B = create_mask_from_id(pic_id, data_path=self.data_path, warp=warp, showGUI=showGUI).convert("RGB")
        B = im_resize(B)

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': A_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
