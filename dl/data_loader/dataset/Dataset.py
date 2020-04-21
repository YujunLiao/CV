from os.path import join, dirname
#from trainer_utils.data_loader.helper.JigsawLoader import JigsawDataset, JigsawTestDataset
# from trainer_utils.data_loader.helper.concat_dataset import ConcatDataset
# from trainer_utils.data_loader.helper.data_helper import Subset
from random import sample
import torch
import warnings

#from torch.utils.data import Dataset
#from trainer_utils.data_loader.helper.JigsawLoader import JigsawTestDatasetMultiple
#from trainer_utils.data_loader.DG_rotation_dataset.RotationDataset import RotationTestDataset, RotationTrainDataset

mnist = 'mnist'
mnist_m = 'mnist_m'
svhn = 'svhn'
synth = 'synth'
usps = 'usps'

vlcs_datasets = ["CALTECH", "LABELME", "PASCAL", "SUN"]
pacs_datasets = ["art_painting", "cartoon", "photo", "sketch"]
office_datasets = ["amazon", "dslr", "webcam"]
digits_datasets = [mnist, mnist, svhn, usps]
available_domains = office_datasets + pacs_datasets + vlcs_datasets + digits_datasets
#office_paths = {dataset: "/home/enoon/data/images/office/%s" % dataset for dataset in office_datasets}
#pacs_paths = {dataset: "/home/enoon/data/images/PACS/kfold/%s" % dataset for dataset in pacs_datasets}
#vlcs_paths = {dataset: "/home/enoon/data/images/VLCS/%s/test" % dataset for dataset in pacs_datasets}
#paths = {**office_paths, **pacs_paths, **vlcs_paths}

dataset_std = {mnist: (0.30280363, 0.30280363, 0.30280363),
               mnist_m: (0.2384788, 0.22375608, 0.24496263),
               svhn: (0.1951134, 0.19804622, 0.19481073),
               synth: (0.29410212, 0.2939651, 0.29404707),
               usps: (0.25887518, 0.25887518, 0.25887518),
               }

dataset_mean = {mnist: (0.13909429, 0.13909429, 0.13909429),
                mnist_m: (0.45920207, 0.46326601, 0.41085603),
                svhn: (0.43744073, 0.4437959, 0.4733686),
                synth: (0.46332872, 0.46316052, 0.46327512),
                usps: (0.17025368, 0.17025368, 0.17025368),
                }

def split(list_1, list_2, proportion=None):
    """

    :param list_1: list of images paths
    :param list_2:  list of labels
    :param proportion: 0 < float < 1
    :return:
    """
    if proportion == None:
        return list_1, list_2
    n = len(list_1)
    # indices is a list of index of (n * percent) samples from original data list.
    indices = sample(range(n), int(n * proportion))
    list_1_1 = [list_1[k] for k in indices]
    list_1_2 = [v for k, v in enumerate(list_1) if k not in indices]
    list_2_1 = [list_2[k] for k in indices]
    list_2_2 = [v for k, v in enumerate(list_2) if k not in indices]
    return [list_1_2, list_1_1, list_2_2, list_2_1]

def paths_and_labels_from(files):
    if isinstance(files, str):
        files = [files]
    paths = []
    labels = []
    for file in files:
        with open(file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            paths.append(line[0])
            labels.append(int(line[1]))
    return [paths, labels]



class MyDataset:
    """Return train, validation, test datasets.

    Implementation:
        train_dataset:Content: train_data_paths, train_labels
        validation_dataset:Content: validation_data_paths, validation_labels
        test_dataset

    """
    def __init__(self, my_training_arguments, is_patch_based_or_not):
        self.train_dataset, self.validation_dataset = self._get_train_and_validation_dataset(my_training_arguments, is_patch_based_or_not)
        self.test_dataset = self._get_test_dataset(my_training_arguments, is_patch_based_or_not)

    def _get_train_and_validation_dataset(self, my_training_arguments, is_patch_based_or_not=False):
        """Return train and validation dataset

        :param my_training_arguments:
        :param is_patch_based_or_not:
        :return:
            train_dataset:Content: train_data_paths, train_labels
            validation_dataset:Content: validation_data_paths, validation_labels
        """
        training_arguments = my_training_arguments.args

        train_dataset_list = []
        validation_dataset_list = []
        max_number_of_train_dataset = training_arguments.limit_source
        source_domains_names_list = training_arguments.source
        assert isinstance(source_domains_names_list, list)

        whole_train_data_paths = []
        whole_validation_data_paths = []
        whole_train_labels = []
        whole_validation_labels = []
        for domain_name in source_domains_names_list:

            # path_of_txt_list_of_data refer to where are the txt files that record all images' path, for example,
            # /home/lyj/Files/project/pycharm/DG_rotation/trainer_utils/data_loader/txt_lists/photo_train.txt
            path = join(dirname(__file__), 'txt_lists', '%s_train.txt' % domain_name)
            data_paths, labels = self.paths_and_labels_from(path)
            # training_arguments.val_size refer to the percent of validation dataset, for example,
            # val_size=0.1 means 10% data is used for validation, 90% data is used for training.
            train_data_paths, validation_data_paths, train_labels, validation_labels = self.split(
                data_paths,
                labels,
                training_arguments.val_size
            )

            whole_train_data_paths += train_data_paths
            whole_validation_data_paths += validation_data_paths
            whole_train_labels += train_labels
            whole_validation_labels += validation_labels


        train_dataset = {'train_data_paths':whole_train_data_paths, 'train_labels':whole_train_labels}
        validation_dataset = {'validation_data_paths':whole_validation_data_paths, 'validation_labels':whole_validation_labels}

            #img_transformer, tile_transformer = get_train_transformers(training_arguments)

            # train_dataset = JigsawDataset(
            #     train_data_paths,
            #     train_labels,
            #     is_patch_based_or_not=is_patch_based_or_not,
            #     img_transformer=img_transformer,
            #     tile_transformer=tile_transformer,
            #     jig_classes=4,
            #     percent_of_original_image=training_arguments.bias_whole_image
            # )

            # train_dataset = RotationTrainDataset(
            #     train_data_paths,
            #     train_labels,
            #     is_patch_based_or_not=is_patch_based_or_not,
            #     img_transformer=img_transformer,
            #     tile_transformer=tile_transformer,
            #     percent_of_original_image=training_arguments.bias_whole_image
            # )
            #
            # if max_number_of_train_dataset:
            #     train_dataset = Subset(train_dataset, max_number_of_train_dataset)
            #
            # train_dataset_list.append(train_dataset)

            # validation_dataset_list.append(
            #     JigsawTestDataset(
            #         validation_data_paths,
            #         validation_labels,
            #         img_transformer=get_val_transformer(training_arguments),
            #         is_patch_based_or_not=is_patch_based_or_not,
            #         jig_classes=4)
            # )

        #     validation_dataset_list.append(
        #         RotationTestDataset(
        #             validation_data_paths,
        #             validation_labels,
        #             img_transformer=get_val_transformer(training_arguments),
        #             is_patch_based_or_not=is_patch_based_or_not,
        #             )
        #     )
        #
        # train_dataset = ConcatDataset(train_dataset_list)
        # validation_dataset = ConcatDataset(validation_dataset_list)
        return train_dataset, validation_dataset

    def _get_test_dataset(self, my_training_arguments, is_patch_based_or_not):
        training_arguments = my_training_arguments.args

        path_of_txt_list = join(dirname(__file__), 'txt_lists', '%s_test.txt' % training_arguments.target)
        data_paths, labels = self.paths_and_labels_from(path_of_txt_list)
        return {'test_data_paths':data_paths, 'test_labels':labels}


def get_dataset(source_domains, target_domain,
                dir=f'{dirname(__file__)}/txt_lists/', val_size = 0.1):
    assert isinstance(source_domains, list)
    assert isinstance(target_domain, str)

    train_paths = []
    val_paths = []
    train_labels = []
    val_labels = []

    for domain in source_domains:
        paths, labels = paths_and_labels_from(f'{dir}/train/{domain}')
        # training_arguments.val_size refer to the percent of validation dataset, for example,
        # val_size=0.1 means 10% data is used for validation, 90% data is used for training.
        paths_1, paths_2, labels_1, labels_2 = split(
            paths,
            labels,
            val_size)
        train_paths += paths_1
        val_paths += paths_2
        train_labels += labels_1
        val_labels += labels_2

    test_paths, test_labels = paths_and_labels_from(f'{dir}/train/{target_domain}')

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels









