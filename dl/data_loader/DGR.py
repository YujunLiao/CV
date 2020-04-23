import torch
from torch.utils.data import DataLoader
from dl.dataset.get_data import get_data
from dl.dataset.rotation import Rotation


train_DL_fn = lambda DS, bs:torch.utils.data.DataLoader(
        DS,
        batch_size=bs,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True)

test_DL_fn = lambda DS, bs:torch.utils.data.DataLoader(
        DS,
        batch_size=bs,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False)


def get_DGR_data_loader(sources='', target='', data_dir='', val_size=float(0),
                        prob=float(0), batch_size=128):
    train_paths, val_paths, train_labels, val_labels = \
        get_data(sources, dir=data_dir+'train/', val_size=val_size)
    test_paths, _, test_labels, _ = get_data(target, dir=data_dir+'test/')

    # dataset
    train_DS = Rotation(train_paths, train_labels, prob=prob)
    val_DS = Rotation(val_paths, val_labels, prob=0)
    test_DS = Rotation(test_paths, test_labels, prob=0)

    return train_DL_fn(train_DS, batch_size), test_DL_fn(val_DS, batch_size),\
           test_DL_fn(test_DS, batch_size)















