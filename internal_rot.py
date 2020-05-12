import copy
from os.path import dirname

import numpy
import torch
from PIL import Image
import numpy as np

from dl.utils import vis
from dl.dataset.base import BaseDataset
from dl.dataset.rotation import Rotation, RotTest, Rot
from dl.dataset.internal_rot import InternalRot
from dl.data_loader.utils.get_p_l import get_p_l
from dl.model.caffenet import caffenet
from torchvision import transforms as tf
from dl.dataset.tf_fn import norm_tf_fn


from dl.data_loader.utils.ds2dl import test_DL_fn

class Container:
    def __init__(self):
        pass

#args = Container()
#args.image_size = 222
# model = caffenet(num_usv_classes=4, num_classes=7)
# model.load_state_dict(wandb.restore('model.pkl', run_path='yujun-liao/DG_rot_caffenet/3t9xfz7c'))
#device = 'cpu'
#print('what', dirname(__file__))
# model.load_state_dict(torch.load('/home/lyj/Files/project/pycharm/CV/data/cache/model.pkl', map_location=device))




paths_1, paths_2, labels_1, labels_2 = get_p_l()
ds = BaseDataset(paths_1, labels_1)


def norm_img(img):
    img_t = tf.ToTensor()(img)
    img_t = norm_tf_fn(img_t)
    return tf.ToPILImage()(img_t)

def distance(i1, i2):
    i1 = tf.ToTensor()(i1)
    i1 = norm_tf_fn(i1)
    i1 = i1.numpy()

    i2 = tf.ToTensor()(i2)
    i2 = norm_tf_fn(i2)
    i2 = i2.numpy()

    return numpy.mean((i1-i2)**2)


def dis_array(indices_1, indices_2, ds=ds):
    n = max(max(indices_1), max(indices_2))+1
    arr = numpy.zeros((n, n)) - 1
    for i in indices_1:
        for j in indices_2:
            img_1, label_1= ds[i]
            img_2, label_2 = ds[j]
            arr[i][j] = distance(img_1, img_2)
    return arr


def get_indices(ds, category):
    return [i for i in range(len(ds)) if ds[i][1]==category]


def show(ds, i_1, i_2):
    img_1, label_1 = ds[i_1]
    img_2, label_2 = ds[i_2]

    img = Image.new('RGB', (img_1.size[0], 2*img_1.size[0]))
    img.paste(img_1, (0, 0))
    img.paste(img_2, (0, img_1.size[0]))
    img.show()


def get_ds(img):
    _ds = [(img, 0)]
    # _ds = [(norm_img(img), 0)]
    for n in range(1, 25):
        global_rot = int((n-1)/6)
        patches = int((n-1)/3)%2
        local_rot = (n-1)%3

        img_ = img.transpose(global_rot+1) if global_rot != 0 else img
        wide = int(img.size[0]/2)
        imgs = []
        for x in [0, wide]:
            for y in [0, wide]:
                imgs.append(img_.crop((x, y, x+wide, y+wide)))
        if patches == 0:
            imgs[0] = imgs[0].transpose(local_rot + 2)
            imgs[3] = imgs[3].transpose(local_rot + 2)
        else:
            imgs[1] = imgs[1].transpose(local_rot + 2)
            imgs[2] = imgs[2].transpose(local_rot + 2)

        for i in range(4):
            # imgs[i] = norm_img(imgs[i])
            pass

        img2 = Image.new('RGB', img.size)
        img2.paste(imgs[0], (0, 0))
        img2.paste(imgs[1], (0, wide))
        img2.paste(imgs[2], (wide, 0))
        img2.paste(imgs[3], (wide, wide))

        _ds.append((img2, n))

    return _ds


# n = np.random.randint(0, len(indices_1)-1)
# print('n=', n)
img = ds[0][0]
indices_2 = range(25)
_ds = get_ds(img)
arr = dis_array(indices_2, indices_2, ds=_ds)

# def s(arr, indices):
#     indices_2 = copy.deepcopy(indices)
#     sum = 0
#     for i in indices:
#         indices_2.remove(i)
#         for j in indices_2:
#             sum += arr[i][j]
#     return sum
#
# s(arr, [0,1,2])




def search(arr, _max, cur_sum, cur_indices, new_i):
    if len(cur_indices) == _max:
        n = (_max**2-_max)/2
        result[cur_sum/n] = copy.deepcopy(cur_indices)
        return

    for i in range(new_i, len(arr)):
        inc = 0
        for j in cur_indices:
            inc += arr[i][j]
        cur_sum += inc
        cur_indices.append(i)
        search(arr, _max, cur_sum, cur_indices, i+1)
        cur_sum -= inc
        cur_indices.remove(i)


cur_indices = []
result = dict()
search(arr, 12, 0, cur_indices, 0)
for key in sorted(result.keys(), reverse=True)[:5]:
    print(key, result[key])
print()
# def search(arr):
#     for i in

# top_num = 3
# indices_of_max_elemants = np.argpartition(arr.flatten(), -top_num)[-top_num:]
# indices_of_max_elemants = np.vstack(np.unravel_index(indices_of_max_elemants, arr.shape)).T
#
# for i_1, i_2 in indices_of_max_elemants:
#     # show(_ds, i_1, i_2)
#     # if ds[i_1][1] == ds[i_2][1]:
#     #     same += 1
#     # else:
#     #     diff += 1
#     print(arr[i_1][i_2])
#     print('label', _ds[i_1][1], _ds[i_2][1])
# print()
#
#
#
#
#
#








