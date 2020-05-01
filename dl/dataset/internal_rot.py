from .base import BaseDataset
from .base import BaseDataset
from.tf_fn import train_tf_fn, test_tf_fn, tile_tf_fn, norm_tf_fn, to_t_tf_fn, to_i_tf_fn
import numpy as np
import torch
import torchvision.transforms as tf


def filter_mat(size=227, margin=10, inside=1):
    """A tensor with 0 on the border and 1 inside.

    :param inside:
    :param size:
    :param margin:
    :return:
    """
    mat = torch.zeros((size, size))
    mat[margin:-margin, margin:-margin] = 1
    if inside == 0:
        mat = torch.ones((size, size)) - mat
    return mat


class InternalRot(BaseDataset):
    def __init__(self, paths='', labels='', prob=float(0),
                 _max=-1, img_size=222, margin=20):
        super().__init__(paths, labels, _max)
        self.prob = prob
        self.img_size = img_size
        self.margin = margin

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        img = tf.Resize((self.img_size, self.img_size))(img)
        img, n = InternalRot.rotate(img, prob=self.prob, margin=self.margin)
        return to_t_tf_fn(img), n, label

    @staticmethod
    def rotate(img, prob=float(0), margin=20):
        """

        :param margin:
        :param img: <class 'PIL.Image'> image
        :param prob: probability of original image
        :return:<class 'PIL.Image'> rotated image

        """
        img_t = tf.ToTensor()(img)
        img_in_t = img_t * filter_mat(size=img_t.shape[1],
                                      margin=margin, inside=1)
        img_out_t = img_t * filter_mat(size=img_t.shape[1],
                                       margin=margin, inside=0)

        img_in = tf.ToPILImage()(img_in_t)
        p_ = (1-prob)/3
        n = np.random.choice(np.arange(4), p=[prob, p_, p_, p_])
        if n != 0:
            img_in = img_in.transpose(n+1)

        img_t = tf.ToTensor()(img_in) + img_out_t
        img = tf.ToPILImage()(img_t)

        return img, n


class InternalRotTrain(InternalRot):
    """Return tensor image with resize, normalize and other transform.

    """
    def __init__(self, paths='', labels='', prob=float(0), _max=-1,
                 args=None, img_size=222, margin=20):
        super().__init__(paths, labels, prob, _max, img_size, margin)
        self.args = args

    def __getitem__(self, index):
        img_t, n, label = super().__getitem__(index)
        img = to_i_tf_fn(img_t)
        img = train_tf_fn(self.args)(img)
        img = tile_tf_fn(self.args)(img)
        img_t = to_t_tf_fn(img)
        img_t = norm_tf_fn(img_t)
        return img_t, n, label


class InternalRotTest(InternalRot):
    """Return tensor image with resize and normalize.

    """
    def __init__(self, paths='', labels='', prob=float(0), _max=-1,
                 args=None, img_size=222, margin=20):
        super().__init__(paths, labels, prob, _max, img_size, margin)
        self.args = args

    def __getitem__(self, index):
        img_t, n, label = super().__getitem__(index)
        img = to_i_tf_fn(img_t)
        img = test_tf_fn(self.args)(img)
        img_t = to_t_tf_fn(img)
        img_t = norm_tf_fn(img_t)
        return img_t, n, label





