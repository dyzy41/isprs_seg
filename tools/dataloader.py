from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from tools import utils

class IsprsSegmentation(Dataset):
    """
    PascalVoc dataset
    """

    def __init__(self,
                 base_dir=None,
                 split='train',
                 transform=None
                 ):
        self.split = split
        self._base_dir = base_dir

        # self.images = os.listdir(os.path.join(self._base_dir, 'image_train'))
        # self.images = [os.path.join(self._base_dir, 'image_train', i) for i in self.images]
        # self.categories = [i.replace('image_train', 'label_train') for i in self.images]

        self.images = os.listdir(os.path.join(self._base_dir, 'image_{}'.format(self.split)))
        self.images = [os.path.join(self._base_dir, 'image_{}'.format(self.split), i) for i in self.images]
        self.images = [i for i in self.images if i.endswith('.tif')]
        self.categories = [i.replace('image_{}'.format(self.split), 'label_{}'.format(self.split)) for i in self.images]

        self.transform = transform
        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.images)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        _img, _target, _name = self._make_img_gt_point_pair(index)
        sample = {'image': _img, 'gt': _target}

        if self.transform is not None:
            sample = self.transform(sample)
        sample['name'] = _name
        return sample

    def _make_img_gt_point_pair(self, index):
        # Read Image and Target
        # _img = np.asarray(Image.open(os.path.join(self.images[index])).convert('RGB')).astype(np.float32)
        # _target = np.asarray(Image.open(os.path.join(self.categories[index])).convert('L')).astype(np.int32)
        _img = utils.read_image(os.path.join(self.images[index])).astype(np.float32)
        _target = utils.read_image(os.path.join(self.categories[index]), 'gt').astype(np.int32)
        # _img = io.read_image(os.path.join(self._image_dir, self.images[index]), driver = 'GDAL')
        # _target = np.asarray(Image.open(os.path.join(self._cat_dir, self.categories[index])).convert('L')).astype(
        #     np.int32)
        return _img, _target, os.path.join(self.images[index]).replace('\\', '/').split('/')[-1]

    def __str__(self):
        return 'VOC2012(split=' + str(self.split) + ')'
