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
            path_of_txt_list = join(dirname(__file__), 'txt_lists', '%s_train.txt' % domain_name)
            data_paths, labels = self._get_data_paths_and_labels_from_txt_list(path_of_txt_list)
            # training_arguments.val_size refer to the percent of validation dataset, for example,
            # val_size=0.1 means 10% data is used for validation, 90% data is used for training.
            train_data_paths, validation_data_paths, train_labels, validation_labels = self._split_dataset_randomly(
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


        # path_of_txt_list_of_data refer to where are the txt files that record all images' path, for example,
        # /home/lyj/Files/project/pycharm/DG_rotation/trainer_utils/data_loader/txt_lists/photo_train.txt
        path_of_txt_list = join(dirname(__file__), 'txt_lists', '%s_test.txt' % training_arguments.target)
        data_paths, labels = self._get_data_paths_and_labels_from_txt_list(path_of_txt_list)

        # img_tr = get_val_transformer(training_arguments)
        # test_dataset = JigsawTestDataset(
        #     data_paths,
        #     labels,
        #     is_patch_based_or_not=is_patch_based_or_not,
        #     img_transformer=img_tr,
        #     jig_classes=4)

        # test_dataset = RotationTestDataset(
        #     data_paths,
        #     labels,
        #     is_patch_based_or_not=is_patch_based_or_not,
        #     img_transformer=img_tr,
        #     )
        #
        # if max_number_of_test_dataset and len(test_dataset) > max_number_of_test_dataset:
        #     test_dataset = Subset(test_dataset, max_number_of_test_dataset)
        #
        #     print("Using %d subset of val dataset" % training_arguments.limit_target)
        # test_dataset_list = [test_dataset]
        # return ConcatDataset(test_dataset_list)
        return {'test_data_paths':data_paths, 'test_labels':labels}

    def _get_data_paths_and_labels_from_txt_list(self, txt_labels):
        with open(txt_labels, 'r') as f:
            lines = f.readlines()

        data_paths = []
        labels = []
        for line in lines:
            line = line.split(' ')
            data_paths.append(line[0])
            labels.append(int(line[1]))

        return data_paths, labels

    def _split_dataset_randomly(self, data_paths, labels, percent):
        """

        :param data_paths: list of images paths
        :param labels:  list of labels
        :param percent: 0 < float < 1
        :return:
        """
        number_of_data = len(data_paths)
        # random_indexes is a list of index of (number_of_data * percent) samples from original data list.
        random_indexes = sample(range(number_of_data), int(number_of_data * percent))
        validation_data_paths = [data_paths[k] for k in random_indexes]
        train_data_paths = [v for k, v in enumerate(data_paths) if k not in random_indexes]
        validation_labels = [labels[k] for k in random_indexes]
        train_labels = [v for k, v in enumerate(labels) if k not in random_indexes]
        return train_data_paths, validation_data_paths, train_labels, validation_labels


