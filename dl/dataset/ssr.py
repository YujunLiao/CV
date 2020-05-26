from math import floor
from .base import BaseDataset
from .base import BaseDataset
from .rotation import Rotation
from.tf_fn import train_tf_fn, test_tf_fn, tile_tf_fn, norm_tf_fn, to_t_tf_fn, to_i_tf_fn
import numpy as np
import torch
import torchvision.transforms as tf
import torchvision.utils as vutils
from PIL import Image


def norm_img(img):
    img_t = to_t_tf_fn(img)
    img_t = norm_tf_fn(img_t)
    return to_i_tf_fn(img_t)



class SemanticSensitiveRot(BaseDataset):
    def __init__(self, paths='', labels='', prob=float(0),
                 _max=-1):
        super().__init__(paths, labels, _max)
        self.prob = prob

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        # img, n = SemanticSensitiveRot.rotate(img, prob=self.prob)
        img, n = SemanticSensitiveRot.deep_all(img, prob=self.prob)
        #img, n = Rotation.rotate(img, prob=self.prob)
        return to_t_tf_fn(img), n, label

    @staticmethod
    def rotate(img, prob=float(0)):
        """24 kinds of rotation out 24

        :param img: <class 'PIL.Image'> image
        :param prob: probability of original image
        :return:<class 'PIL.Image'> rotated image,

        """
        p_ = (1-prob)/24
        n = np.random.choice(np.arange(25), p=[prob]+[p_ for i in range(24)])

        if n == 0:
            return img, 0

        global_rot = int((n-1)/6)
        patches = int((n-1)/3)%2
        local_rot = (n-1)%3

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

        for i in range(4):
            imgs[i] = norm_img(imgs[i])

        img2 = Image.new('RGB', img.size)
        img2.paste(imgs[0], (0, 0))
        img2.paste(imgs[1], (0, wide))
        img2.paste(imgs[2], (wide, 0))
        img2.paste(imgs[3], (wide, wide))

        return img2, n

    @staticmethod
    def rotate2(img, prob=float(0)):
        """24 kinds of rotation out 2

        :param img: <class 'PIL.Image'> image
        :param prob: probability of original image
        :return:<class 'PIL.Image'> rotated image,

        """
        p_ = (1-prob)/24
        n = np.random.choice(np.arange(25), p=[prob]+[p_ for i in range(24)])

        if n == 0:
            return img, 0

        global_rot = int((n-1)/6)
        patches = int((n-1)/3)%2
        local_rot = (n-1)%3

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

        for i in range(4):
            imgs[i] = norm_img(imgs[i])

        img2 = Image.new('RGB', img.size)
        img2.paste(imgs[0], (0, 0))
        img2.paste(imgs[1], (0, wide))
        img2.paste(imgs[2], (wide, 0))
        img2.paste(imgs[3], (wide, wide))

        return img2, patches+1

    @staticmethod
    def rotate3(img, prob=float(0)):
        """24 kinds of rotation out 3

        :param img: <class 'PIL.Image'> image
        :param prob: probability of original image
        :return:<class 'PIL.Image'> rotated image,

        """
        p_ = (1-prob)/24
        n = np.random.choice(np.arange(25), p=[prob]+[p_ for i in range(24)])

        if n == 0:
            return img, 0

        global_rot = int((n-1)/6)
        patches = int((n-1)/3)%2
        local_rot = (n-1)%3

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

        for i in range(4):
            imgs[i] = norm_img(imgs[i])

        img2 = Image.new('RGB', img.size)
        img2.paste(imgs[0], (0, 0))
        img2.paste(imgs[1], (0, wide))
        img2.paste(imgs[2], (wide, 0))
        img2.paste(imgs[3], (wide, wide))

        return img2, local_rot+1

    @staticmethod
    def rotate4(img, prob=float(0)):
        """24 kinds of rotation out 3

        :param img: <class 'PIL.Image'> image
        :param prob: probability of original image
        :return:<class 'PIL.Image'> rotated image,

        """
        p_ = (1-prob)/24
        n = np.random.choice(np.arange(25), p=[prob]+[p_ for i in range(24)])

        if n == 0:
            return img, 0

        global_rot = int((n-1)/6)
        patches = int((n-1)/3)%2
        local_rot = (n-1)%3

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

        for i in range(4):
            imgs[i] = norm_img(imgs[i])

        img2 = Image.new('RGB', img.size)
        img2.paste(imgs[0], (0, 0))
        img2.paste(imgs[1], (0, wide))
        img2.paste(imgs[2], (wide, 0))
        img2.paste(imgs[3], (wide, wide))

        return img2, global_rot+1

    @staticmethod
    def rotate_slelct_3(img, prob=float(0)):
        """24 kinds of rotation out 3

        :param img: <class 'PIL.Image'> image
        :param prob: probability of original image
        :return:<class 'PIL.Image'> rotated image,

        """
        p_ = (1 - prob) / 3
        n = np.random.choice([0, 10, 13, 20], p=[prob] + [p_ for i in range(3)])
        indice_dict = {0: 0, 10: 1, 13: 2, 20: 3}

        if n == 0:
            return img, 0

        global_rot = int((n - 1) / 6)
        patches = int((n - 1) / 3) % 2
        local_rot = (n - 1) % 3

        if global_rot != 0:
            img = img.transpose(global_rot + 1)

        wide = int(img.size[0] / 2)
        imgs = []
        for x in [0, wide]:
            for y in [0, wide]:
                imgs.append(img.crop((x, y, x + wide, y + wide)))
        if patches == 0:
            imgs[0] = imgs[0].transpose(local_rot + 2)
            imgs[3] = imgs[3].transpose(local_rot + 2)
        else:
            imgs[1] = imgs[1].transpose(local_rot + 2)
            imgs[2] = imgs[2].transpose(local_rot + 2)

        for i in range(4):
            imgs[i] = norm_img(imgs[i])

        img2 = Image.new('RGB', img.size)
        img2.paste(imgs[0], (0, 0))
        img2.paste(imgs[1], (0, wide))
        img2.paste(imgs[2], (wide, 0))
        img2.paste(imgs[3], (wide, wide))


        return img2, indice_dict[n]

    @staticmethod
    def deep_all(img, prob=float(0)):
        return img, 0



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
        if n==0:
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





