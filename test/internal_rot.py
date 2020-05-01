from os.path import dirname
import torch
from PIL import Image
import numpy as np

from dl.utils import vis
from dl.dataset.rotation import Rotation, RotTest, Rot
from dl.dataset.internal_rot import InternalRot
from dl.data_loader.utils.get_p_l import get_p_l
from dl.model.caffenet import caffenet
from torchvision import transforms as tf


from dl.data_loader.utils.ds2dl import test_DL_fn

class Container:
    def __init__(self):
        pass

args = Container()
args.image_size = 222
model = caffenet(num_usv_classes=4, num_classes=7)
# model.load_state_dict(wandb.restore('model.pkl', run_path='yujun-liao/DG_rot_caffenet/3t9xfz7c'))
device = 'cpu'
print('what', dirname(__file__))
model.load_state_dict(torch.load('/home/lyj/Files/project/pycharm/CV/data/cache/model.pkl', map_location=device))
paths_1, paths_2, labels_1, labels_2 = get_p_l()
ds = InternalRot(paths_1, labels_1, prob=0)

for i in range(10):
    img_t = ds[i][0]
    tf.ToPILImage()(img_t).show()
    print()




def filter_mat(size=227, margin=10, inside=1):
    """A tensor with 0 on the border and 1 inside.

    :param size:
    :param margin:
    :return:
    """
    mat = torch.zeros((size, size))
    mat[margin:-margin, margin:-margin] = 1
    if inside == 0:
        mat = torch.ones((size, size)) - mat
    return mat


img_in_t = img_t * filter_mat(size=img_t.shape[1], inside=1)
img_out_t = img_t * filter_mat(size=img_t.shape[1], inside=0)

img_in = tf.ToPILImage()(img_in_t)
img_in_r, n = Rotation.rotate(img_in, prob=0)
img_t = tf.ToTensor()(img_in_r) + img_out_t
img = tf.ToPILImage()(img_t)
img.show()
print()

# dl = test_DL_fn(ds, 10)
#
#
# for i, (imgs_t, _, _) in enumerate(dl):
#
#     print()


