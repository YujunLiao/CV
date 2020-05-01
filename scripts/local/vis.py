from os.path import dirname
import torch

from dl.utils import vis
from dl.dataset.rotation import Rotation, RotTest
from dl.data_loader.utils.get_p_l import get_p_l
from dl.model.caffenet import caffenet
from torchvision.transforms import transforms as tf
from torchvision import utils as vutils

import wandb
from dl.data_loader.utils.ds2dl import test_DL_fn

class Container:
    def __init__(self):
        pass
wandb.init()
args = Container()
args.image_size = 222
model = caffenet(num_usv_classes=4, num_classes=7)
# model.load_state_dict(wandb.restore('model.pkl', run_path='yujun-liao/DG_rot_caffenet/3t9xfz7c'))
device = 'cpu'
print('what', dirname(__file__))
model.load_state_dict(torch.load('/home/lyj/Files/project/pycharm/CV/data/cache/model.pkl', map_location=device))

paths_1, paths_2, labels_1, labels_2 = get_p_l()
ds = RotTest(paths_1, labels_1, prob=1, args=args)
dl = test_DL_fn(ds, 10)

for i, (imgs_t, _, _) in enumerate(dl):
    img_grid = vutils.make_grid(imgs_t, normalize=True, scale_each=True, nrow=1)
    wandb.log({'original_img': [wandb.Image(tf.ToPILImage()(img_grid))]})
    model.eval()
    vis.feature_map(model, imgs_t, wandb)
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