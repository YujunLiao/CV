# from torchvision import transforms
import torch
from torch.utils.data import DataLoader
# from trainer_utils.data_loader.helper.JigsawLoader import get_split_dataset_info
# from os.path import join, dirname
# from trainer_utils.data_loader.helper.JigsawLoader import JigsawDataset, JigsawTestDataset, get_split_dataset_info, _dataset_info
# from trainer_utils.data_loader.helper.concat_dataset import ConcatDataset
# from random import sample, random
from trainer_utils.data_loader.DG_rotation_dataset.DGRotationDataset import DARotationDataset


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

    # def _get_test_data_loader(self, my_training_arguments, is_patch_based_or_not=False):
    #     args = my_training_arguments.training_arguments
    #     names, labels = _dataset_info(join(dirname(__file__), 'txt_lists', '%s_test.txt' % args.target))
    #     img_tr = get_val_transformer(args)
    #     test_dataset_list = JigsawTestDataset(names, labels, patches=is_patch_based_or_not, img_transformer=img_tr, jig_classes=4)
    #     if args.limit_target and len(test_dataset_list) > args.limit_target:
    #         test_dataset_list = Subset(test_dataset_list, args.limit_target)
    #         print("Using %d subset of val dataset" % args.limit_target)
    #     test_dataset = ConcatDataset([test_dataset_list])
    #
    #     test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4,
    #                                          pin_memory=True, drop_last=False)
    #     return test_data_loader







