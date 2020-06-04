import json
from pathlib import Path

from data.aligned_dataset import AlignedDataset
from data.base_dataset import get_params, get_transform
from PIL import Image


class UnalignedDomainDataset(AlignedDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test' containing also the coordinates.txt file.
    As well as a directory '/path/to/unaligned'
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        AlignedDataset.modify_commandline_options(parser=parser, is_train=is_train)
        parser.add_argument('--unaligned_dir', help='Unaligned dir')

        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        # TODO: load json file with data

        assert self.opt.no_flip is True, "No flipping allowed"
        assert self.opt.preprocess == "none", "No preprocessing allowed"

        with open(opt.dataroot + "/aligned/all_coordinates.txt", 'r') as f:
            self.json_data = json.load(f)

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

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        # get unaligned inputs
        filename = Path(AB_path).name
        unaligned_path = str(Path(self.opt.unaligned_dir) / filename[:3] / filename[6:])

        # cut
        unaligned_B = Image.open(unaligned_path)
        unaliged_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1), no_resize=True)
        unaligned_B = unaliged_transform(unaligned_B)

        # get params
        alignement_params = self.json_data[filename[:-4]]['alignment_params']

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path,
                'unaligned_B': unaligned_B, 'unaligned_path': unaligned_path,
                'alignment_params': alignement_params,
                }

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
