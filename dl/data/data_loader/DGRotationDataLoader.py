# from torchvision import transforms
import torch
from torch.utils.data import DataLoader
# from trainer_utils.data_loader.helper.JigsawLoader import get_split_dataset_info
# from os.path import join, dirname
# from trainer_utils.data_loader.helper.JigsawLoader import JigsawDataset, JigsawTestDataset, get_split_dataset_info, _dataset_info
# from trainer_utils.data_loader.helper.concat_dataset import ConcatDataset
# from random import sample, random
from dl.data.dataset.DGRotationDataset import DARotationDataset
from dl.data.dataset.get_data import get_data

from dl.data.dataset.rotation import Rotation


def get_DG_data_loader(sources='', target='', data_dir='', val_size=float(0),
                       prob=float(0), batch_size=128):
    train_paths, val_paths, train_labels, val_labels = \
        get_data(sources, dir=data_dir+'train/', val_size=val_size)
    test_paths, _, test_labels, _ = get_data(target, dir=data_dir+'test/')

    # dataset
    train_DS = Rotation(train_paths, train_labels, prob=prob)
    val_DS = Rotation(val_paths, val_labels, prob=0)
    test_DS = Rotation(test_paths, test_labels, prob=0)

    train_DL = torch.utils.data.DataLoader(
        train_DS,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    val_DL = torch.utils.data.DataLoader(
        val_DS,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )
    test_DL = torch.utils.data.DataLoader(
        test_DS,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    return train_DL, val_DL, test_DL















