import torch
from torchvision.transforms import transforms as tf
import torchvision.utils as vutils





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

def feature_map(model, imgs_t, wandb, skip_keys='usv'):
        feature_maps = []
        for name, layer in wrap_model(model, skip_keys):
            if 'fc' in name:
                imgs_t = imgs_t.view(imgs_t.size(0), -1)
            imgs_t = layer(imgs_t)
            if 'conv' in name:
                x1 = torch.reshape(imgs_t, (-1, 1, imgs_t.shape[2], imgs_t.shape[3]))
                img_grid = vutils.make_grid(x1, normalize=True, scale_each=True, nrow=imgs_t.shape[1])  # normalize进行归一化处理
                # tf.ToPILImage()(img_grid).show()
                if wandb:
                    feature_maps.append(wandb.Image(tf.ToPILImage()(img_grid), caption=name.replace('.', '-')))
                    wandb.log({name.replace('.', '/'): [wandb.Image(tf.ToPILImage()(img_grid), caption=name.replace('.', '-'))]})



















