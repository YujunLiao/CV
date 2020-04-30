import numpy as np
from .base import BaseDataset
from.tf_fn import train_tf_fn, test_tf_fn, tile_tf_fn, norm_tf_fn

class Rotation(BaseDataset):
    def __init__(self, paths='', labels='', prob=float(0), _max=-1):
        super().__init__(paths, labels, _max)
        self.prob = prob

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        img, n = Rotation.rotate(img, prob=self.prob)
        return img, n, label

    @staticmethod
    def rotate(img, prob=float(0)):
        """

        :param img: <class 'PIL.Image'> image
        :param prob: probability of original image
        :return:<class 'PIL.Image'> rotated image,

        """
        p_ = (1-prob)/3
        n = np.random.choice(np.arange(4), p=[prob, p_, p_, p_])
        if n != 0:
            img = img.transpose(n+1)
        return img, n


class RotTrain(Rotation):
    def __init__(self, paths='', labels='', prob=float(0), _max=-1, args=None):
        super().__init__(paths, labels, prob, _max)
        self.args = args

    def __getitem__(self, index):
        img, n, label = super().__getitem__(index)
        img = train_tf_fn(self.args)(img)
        img = tile_tf_fn(self.args)(img)
        img_t = norm_tf_fn(img)
        return img_t, n, label


class RotTest(Rotation):
    def __init__(self, paths='', labels='', prob=float(0), _max=-1, args=None):
        super().__init__(paths, labels, prob, _max)
        self.args = args

    def __getitem__(self, index):
        img, n, label = super().__getitem__(index)
        img = test_tf_fn(self.args)(img)
        img_t = norm_tf_fn(img)
        return img_t, n, label


