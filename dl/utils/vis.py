import os
import torch
from dl.dataset.rotation import Rotation, RotTest
from dl.data_loader.utils.get_p_l import get_p_l
from dl.model.caffenet import caffenet
from torchvision.transforms import transforms as tf
import torchvision.utils as vutils
import wandb
import torch.nn.functional as F
import torch.nn as nn
from dl.data_loader.utils.ds2dl import test_DL_fn


class Container:
    def __init__(self):
        pass

def wrap_model(model, skip_keys):
    if isinstance(skip_keys, str):
        skip_keys = [skip_keys]
    named_layers = []
    for name, layer in model.named_modules():
        if name == '':
            continue
        flag = False
        for key in skip_keys:
            if key in name:
                flag = True
        if flag:
            continue
        if isinstance(layer, torch.nn.Sequential):
            continue
        named_layers.append((name, layer))
    return named_layers

def feature_map(model, imgs_t, wandb):
        feature_maps = []
        for name, layer in wrap_model(model, 'usv'):
            if 'fc' in name:
                imgs_t = imgs_t.view(imgs_t.size(0), -1)
            imgs_t = layer(imgs_t)
            if 'conv' in name:
                x1 = imgs_t.transpose(0, 1)  # C，B, H, W  ---> B，C, H, W
                img_grid = vutils.make_grid(x1, normalize=True, scale_each=True, nrow=4)  # normalize进行归一化处理
                if wandb:
                    feature_maps.append(wandb.Image(tf.ToPILImage()(img_grid), caption=name.replace('.', '-')))
        if wandb:
            wandb.log({'feature_map': feature_maps})



wandb.init()
args = Container()
args.image_size = 222
model = caffenet(num_usv_classes=4, num_classes=7)
paths_1, paths_2, labels_1, labels_2 = get_p_l()
ds = RotTest(paths_1, labels_1, prob=1)
dl = test_DL_fn(ds, 4)

for i, (imgs_t, _, _) in enumerate(dl):
    model.eval()
    wandb.log({'original_img': [wandb.Image(tf.ToPILImage()(i)) for i in imgs_t]})
    feature_map(model, imgs_t, wandb)
    break






# for i, (name, param) in enumerate(model.named_parameters()):
#     if 'conv' in name and 'weight' in name:
#         in_channels = param.size()[1]
#         out_channels = param.size()[0]   # 输出通道，表示卷积核的个数
#
#         k_w, k_h = param.size()[3], param.size()[2]   # 卷积核的尺寸
#         kernel_all = param.view(-1, 1, k_w, k_h)  # 每个通道的卷积核
#         kernel_grid = vutils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=in_channels)
#         img = tf.ToPILImage()(kernel_grid)
#         # img.show()
#         wandb.log({"examples": [wandb.Image(img, caption="Label")]})
#         # for i in range(len(kernel_all)):
#         #     img = tf.ToPILImage()(kernel_all[i])
#         #     img.show()









