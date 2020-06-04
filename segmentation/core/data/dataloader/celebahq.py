"""Prepare CelebAHQ dataset"""
import os
import torch
import numpy as np

from PIL import Image
from .segbase import SegmentationDataset


class CelebaHQSegmentation(SegmentationDataset):
    NUM_CLASS = 15
    def __init__(self, root='', split='train', mode=None, transform=None, **kwargs):
        super(CelebaHQSegmentation, self).__init__(root, split, mode, transform, **kwargs)
        assert os.path.exists(self.root), "Please setup the dataset in segmentation/core/data/dataloader/celebahq.py"
        self.images = _get_aligned_images(self.root, file_type='.png')

        if len(self.images) == 0:
            raise RuntimeError("Found 0 images in subfolders of:" + root + "\n")

        # self.valid_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        self._mapping = np.array([[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204],
                                 [0, 255, 255], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153],
                                 [0, 0, 204], [255, 153, 51], [255, 51, 153]]).astype('int32')
        self._mapping_list = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204],
                         [0, 255, 255], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153],
                          [0, 0, 204], [255, 153, 51], [255, 51, 153]]
        '''
        'skin' 'color': (204, 0, 0)
        'nose': 'color': (76, 153, 0)
        'glasses': 'color': (204, 204, 0)
        'l_eye': 'color': (51, 51, 255)
        'r_eye':  'color': (204, 0, 204)
        'brows':  "_r_brow.png"], 'color': (0, 255, 255)
        'l_ear': 'color': (102, 51, 0)
        'r_ear': 'color': (255, 0, 0)
        'mouth':  'color': (102, 204, 0)
        'u_lip':  'color': (255, 255, 0)
        'l_lip': , 'color': (0, 0, 153)
        'hair': 'color': (0, 0, 204)
        'neck':  'color': (255, 153, 51)
        'misc': ["_hat.png", "_cloth.png", "ear_r.png"], 'color': (255, 51, 153)
        '''

    def _class_to_index(self, mask):
        # assert the value
        # uniques = np.unique(mask, axis=0)
        # for row in uniques:
        #     values = np.unique(row, axis=0)
        #     for value in values:
        #         print(value)
        #         assert (value in self._mapping)

        index = []
        for row in mask:
            for item in row:
                try:
                    i = self._mapping_list.index(item.tolist())
                    index.append(i)
                except ValueError:
                    # due to scaling some mask pixels are not retained?
                    i = find_nearest_index(self._mapping_list, item)
                    index.append(i)
                    pass
        index = np.asarray(index).reshape((mask.shape[0], mask.shape[1]))
        return index

    def __getitem__(self, index):
        AB = Image.open(self.images[index]).convert('RGB')
        w, h = AB.size
        if w != h:
            img = AB.crop((0, 0, w / 2, h))
            mask = AB.crop((w/2, 0, w, h))
        else:
            img = AB
            mask = AB
        if h != 256:
            img = img.resize((256, 256), Image.NEAREST)
            mask = mask.resize((256, 256), Image.NEAREST)
        if self.mode == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img, os.path.basename(self.images[index])

        # synchrosized transform
        if self.mode == 'train':
            img, mask = self._sync_transform(img, mask)
        elif self.mode == 'val':
            img, mask = self._val_sync_transform(img, mask)
        else:
            assert self.mode == 'testval'
            img, mask = self._img_transform(img), self._mask_transform(mask)
        # general resize, normalize and toTensor
        if self.transform is not None:
            img = self.transform(img)
        return img, mask, os.path.basename(self.images[index])

    def _mask_transform(self, mask):
        target = self._class_to_index(np.array(mask).astype('int32'))
        return torch.LongTensor(np.array(target).astype('int32'))

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.images)

    @property
    def pred_offset(self):
        return 0


def _get_aligned_images(img_folder, file_type='.jpg'):
    img_paths = []
    for root, _, files in os.walk(img_folder):
        for filename in files:
            if filename.endswith(file_type):
                imgpath = os.path.join(root, filename)
                if os.path.isfile(imgpath):
                    img_paths.append(imgpath)
                else:
                    print('cannot find the image:', imgpath)
    print('Found {} images in the folder {}'.format(len(img_paths), img_folder))

    return img_paths


def find_nearest_index(array, value):

    array = np.asarray(array)
    idx = np.mean(np.abs(array - value), axis=1).argmin()

    return idx


if __name__ == '__main__':
    dataset = CelebaHQSegmentation()
