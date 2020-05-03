from math import floor
from .base import BaseDataset
from .base import BaseDataset
from.tf_fn import train_tf_fn, test_tf_fn, tile_tf_fn, norm_tf_fn, to_t_tf_fn, to_i_tf_fn
import numpy as np
import torch
import torchvision.transforms as tf
import torchvision.utils as vutils
from PIL import Image


class SemanticSensitiveRot(BaseDataset):
    def __init__(self, paths='', labels='', prob=float(0),
                 _max=-1):
        super().__init__(paths, labels, _max)
        self.prob = prob

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        img, n = SemanticSensitiveRot.rotate(img, prob=self.prob)
        return to_t_tf_fn(img), n, label

    @staticmethod
    def rotate(img, prob=float(0)):
        """

        :param img: <class 'PIL.Image'> image
        :param prob: probability of original image
        :return:<class 'PIL.Image'> rotated image,

        """
        p_ = (1-prob)/24
        p_list = [p_ for i in range(24)]
        p_list.append(prob)
        n = np.random.choice(np.arange(25), p=p_list)

        if n == 24:
            return img, 24

        global_rot = int(n/6)
        patches = int(n/3)%2
        local_rot = n%3

        if global_rot != 0:
            img = img.transpose(global_rot+1)

        wide = int(img.size[0]/2)
        imgs = []
        for x in [0, wide]:
            for y in [0, wide]:
                imgs.append(img.crop((x, y, x+wide, y+wide)))
        if patches == 0:
            imgs[0] = imgs[0].transpose(local_rot + 2)
            imgs[3] = imgs[3].transpose(local_rot + 2)
        else:
            imgs[1] = imgs[1].transpose(local_rot + 2)
            imgs[2] = imgs[2].transpose(local_rot + 2)

        img2 = Image.new('RGB', img.size)
        img2.paste(imgs[0], (0, 0))
        img2.paste(imgs[1], (0, wide))
        img2.paste(imgs[2], (wide, 0))
        img2.paste(imgs[3], (wide, wide))

        return img2, n


class SSRTrain(SemanticSensitiveRot):
    """Return tensor image with resize, normalize and other transform.

    """
    def __init__(self, paths='', labels='', prob=float(0), _max=-1, args=None):
        super().__init__(paths, labels, prob, _max)
        self.args = args

    def __getitem__(self, index):
        img_t, n, label = super().__getitem__(index)
        img = to_i_tf_fn(img_t)
        img = train_tf_fn(self.args)(img)
        img = tile_tf_fn(self.args)(img)
        img_t = to_t_tf_fn(img)
        img_t = norm_tf_fn(img_t)
        return img_t, n, label


class SSRTest(SemanticSensitiveRot):
    """Return tensor image with resize and normalize.

    """
    def __init__(self, paths='', labels='', prob=float(0), _max=-1, args=None):
        super().__init__(paths, labels, prob, _max)
        self.args = args

    def __getitem__(self, index):
        img_t, n, label = super().__getitem__(index)
        img = to_i_tf_fn(img_t)
        img = test_tf_fn(self.args)(img)
        img_t = to_t_tf_fn(img)
        img_t = norm_tf_fn(img_t)
        return img_t, n, label





