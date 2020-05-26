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


def dis_array(indices_1, indices_2, ds):
    n = max(max(indices_1), max(indices_2))+1
    arr = numpy.zeros((n, n)) - 1
    for i in indices_1:
        for j in indices_2:
            img_1, label_1 = ds[i]
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


def get_img_transformed_25(img, n):
    if n == 0:
        return norm_img(img)
    global_rot = int((n - 1) / 6)
    patches = int((n - 1) / 3) % 2
    local_rot = (n - 1) % 3

    img_ = img.transpose(global_rot + 1) if global_rot != 0 else img
    wide = int(img.size[0] / 2)
    imgs = []
    for x in [0, wide]:
        for y in [0, wide]:
            imgs.append(img_.crop((x, y, x + wide, y + wide)))
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
    return img2

def get_ds_25(img):
    _ds = [(img, 0)]
    # _ds = [(norm_img(img), 0)]
    for n in range(1, 25):
        img2 = get_img_transformed_25(img, n)
        _ds.append((img2, n))
    return _ds

def get_ds_28(img):
    _ds = [(img, 0)]
    # _ds = [(norm_img(img), 0)]
    for n in range(1, 25):
        img2 = get_img_transformed_25(img, n)
        _ds.append((img2, n))
    for n in range(3):
        img_2 = img.transpose(n+2)
        _ds.append((img_2, n+25))

    return _ds


def get_ds_r(img):
    _ds = [(img, 0)]
    for n in range(3):
        img_2 = img.transpose(n+2)
        _ds.append((img_2, n+1))
    return _ds


# n = np.random.randint(0, len(indices_1)-1)
# print('n=', n)


def search(arr, _max, cur_sum, cur_indices, new_i):
    if len(cur_indices) == _max:
        n = (_max**2-_max)/2
        if cur_sum/n in result.keys():
            # print(cur_indices, result[cur_sum/n])
            print('*', end='')
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


paths_1, paths_2, labels_1, labels_2 = get_p_l()
ds = BaseDataset(paths_1, labels_1)

# img = ds[100][0]
# get_img_transformed_25(img, 0).show()
# get_img_transformed_25(img, 10).show()
# get_img_transformed_25(img, 13).show()
# get_img_transformed_25(img, 20).show()

n = 25
counter = 0
arr = numpy.zeros((n, n))
for i in np.linspace(0, len(ds)-1, 100, dtype=np.int):
    counter += 1
    img = ds[i][0]
    print(counter, end=' ')
    # img.show()
    indices_2 = range(n)
    _ds = get_ds_25(img)
    arr += dis_array(indices_2, indices_2, ds=_ds)
    n_arr = arr/counter



def s(arr, indices):
    sum = 0
    counter = 0
    for i in range(len(indices)):
        for j in range(i+1, len(indices)):
            sum += arr[indices[i]][indices[j]]
            counter += 1
    return sum/counter

# dis = s(n_arr, [2, 5, 15])
# print(dis)
# print()

# cur_indices = []
# i = 3
# result = dict()
# search(n_arr, i, 0, cur_indices, 0)
# print('now take i images:', i)
# for key in sorted(result.keys(), reverse=True)[:5]:
#     print(key, result[key])
# print()
# for i in range(1, 25):
#     indices = [0, 25, 26, 27]+[i]
#     print(indices, s(n_arr,indices))


cur_indices = []
for i in [4, 5, 6]:
    result = dict()
    search(n_arr, i, 0, cur_indices, 0)
    print('now take i images:', i)
    for key in sorted(result.keys(), reverse=True)[:20]:
        # if 0 not in result[key]:
        #     continue
        print(key, result[key])

    # for key in sorted(result.keys())[:5]:
    #     print(key, result[key])

# for key in sorted(result.keys()):
#     if result[key][0] !=0:
#         print(key, result[key])
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









