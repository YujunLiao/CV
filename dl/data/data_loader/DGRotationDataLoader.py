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

class DGRotationDataLoader:
    """Return train, validation, test data loaders.

    Implementation:
        source_data_loader
        validation_data_loader
        test_data_loader

    """
    def __init__(self, my_training_arguments, is_patch_based_or_not):
        """

        :param my_training_arguments:
        :param is_patch_based_or_not:
        """
        self.train_data_loader, self.validation_data_loader, self.test_data_loader = self._get_train_and_validation_and_test_data_loader(
            my_training_arguments,
            is_patch_based_or_not=is_patch_based_or_not
        )
        # self.test_data_loader = self._get_test_data_loader(
        #     my_training_arguments,
        #     is_patch_based_or_not=is_patch_based_or_not
        # )

    def _get_train_and_validation_and_test_data_loader(self, my_training_arguments, is_patch_based_or_not=False):


        # train_dataset, validation_dataset = self._get_train_and_validation_dataset(
        #     my_training_arguments,
        #     is_patch_based_or_not
        # )

        # my_dataset=MyDataset(my_training_arguments, is_patch_based_or_not)
        # train_dataset = my_dataset.train_dataset
        # validation_dataset = my_dataset.validation_dataset
        # test_dataset = my_dataset.test_dataset

        rotation_dataset = DARotationDataset(my_training_arguments, is_patch_based_or_not)
        train_dataset=rotation_dataset.train_dataset
        validation_dataset=rotation_dataset.validation_dataset
        test_dataset=rotation_dataset.test_dataset

        # dataset =
        # val_dataset = ConcatDataset(validation_dataset_list)
        # TODO(lyj): drop_last
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=my_training_arguments.args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        validation_data_loader = torch.utils.data.DataLoader(
            validation_dataset,
            batch_size=my_training_arguments.args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )

        test_data_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=my_training_arguments.args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            drop_last=False
        )

        return train_data_loader, validation_data_loader, test_data_loader



def get_DG_data_loader(sources='', target='', data_dir='', val_size=float(0),
                       prob=float(0), batch_size=128):
    train_paths, val_paths, train_labels, val_labels = \
        get_data(sources, dir=data_dir+'train/', val_size=val_size)
    test_paths, _, test_labels, _ = get_data(target, dir=data_dir+'test/')

    # dataset
    train_DS = Rotation(train_paths, train_labels, prob=prob)
    val_DS = Rotation(val_paths, val_labels, prob=0)
    test_DS = Rotation(train_paths, test_labels, prob=0)

    f = lambda DS:\
        torch.utils.data.DataLoader(
        DS,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    return [f(_) for _ in [train_DS, val_DS, test_DS]]















