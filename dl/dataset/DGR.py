import numpy as np
#import torch
import torch.utils.data as data
# import torchvision
import torchvision.transforms as transforms
from PIL import Image
from random import random
import torch
import bisect
import warnings


from torch.utils.data import Dataset
from dl.data_loader.utils import get_data


class DARotationDataset():
    def __init__(self, my_training_arguments, is_patch_based_or_not):
        args = my_training_arguments.args
        train_paths, val_paths, train_labels, val_labels = \
        get_data(args.source, dir=args.data_dir+'train/', val_size=args.val_size)
        test_paths, _, test_labels, _ = get_data(args.target, dir=args.data_dir+'test/')
        max_number_of_train_dataset = args.limit_source
        max_number_of_test_dataset = args.limit_target
        img_transformer, tile_transformer = self._get_train_transformers(args)
        val_transformer = self._get_val_transformer(args)

        train_dataset = RotationTrainDataset(
            train_paths,
            train_labels,
            # my_dataset.train_dataset['train_data_paths'],
            # my_dataset.train_dataset['train_labels'],
            is_patch_based_or_not=is_patch_based_or_not,
            img_transformer=img_transformer,
            tile_transformer=tile_transformer,
            percent_of_original_image=args.bias_whole_image
        )

        if max_number_of_train_dataset:
            train_dataset = Subset(train_dataset, max_number_of_train_dataset)

        self.train_dataset = ConcatDataset([train_dataset])



        validation_dataset = RotationTestDataset(
            val_paths,
            val_labels,
            # my_dataset.validation_dataset['validation_data_paths'],
            # my_dataset.validation_dataset['validation_labels'],
            img_transformer=val_transformer,
            is_patch_based_or_not=is_patch_based_or_not,

        )
        self.validation_dataset = ConcatDataset([validation_dataset])




        test_dataset = RotationTestDataset(
            test_paths,
            test_labels,
            # my_dataset.test_dataset['test_data_paths'],
            # my_dataset.test_dataset['test_labels'],
            is_patch_based_or_not=is_patch_based_or_not,
            img_transformer=val_transformer,
            )

        if max_number_of_test_dataset and len(test_dataset) > max_number_of_test_dataset:
            test_dataset = Subset(test_dataset, max_number_of_test_dataset)

            # print("Using %d subset of val dataset" % training_arguments.limit_target)

        self.test_dataset = ConcatDataset([test_dataset])

    def _get_train_transformers(self, training_arguments):
        img_tr = [transforms.RandomResizedCrop((int(training_arguments.image_size), int(training_arguments.image_size)),
                                               (training_arguments.min_scale, training_arguments.max_scale))]
        if training_arguments.random_horiz_flip > 0.0:
            img_tr.append(transforms.RandomHorizontalFlip(training_arguments.random_horiz_flip))
        if training_arguments.jitter > 0.0:
            img_tr.append(transforms.ColorJitter(brightness=training_arguments.jitter, contrast=training_arguments.jitter, saturation=training_arguments.jitter,
                                                 hue=min(0.5, training_arguments.jitter)))

        tile_tr = []
        if training_arguments.tile_random_grayscale:
            tile_tr.append(transforms.RandomGrayscale(training_arguments.tile_random_grayscale))
        tile_tr = tile_tr + [transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

        return transforms.Compose(img_tr), transforms.Compose(tile_tr)

    def _get_val_transformer(self, training_arguments):
        img_tr = [transforms.Resize((training_arguments.image_size, training_arguments.image_size)), transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        return transforms.Compose(img_tr)


class RotationTrainDataset(data.Dataset):
    def __init__(self, data_paths, labels, img_transformer=None, tile_transformer=None, is_patch_based_or_not=False, percent_of_original_image=None):

        self._data_path = ""
        self._data_paths = data_paths
        self._labels = labels
        # the shape of self.permutations is 30*9
        # self.permutations = self.__retrieve_permutations(jig_classes)
        # self.grid_size = 3
        self._bias_whole_image = percent_of_original_image
        if is_patch_based_or_not:
            self.patch_size = 64
        self._image_transformer = img_transformer
        self._augment_tile = tile_transformer
        if is_patch_based_or_not:
            self.returnFunc = lambda x: x
        # else:
        #
        #     def make_grid(x):
        #         """
        #
        #         :param x:
        #         :return:
        #         """
        #         return torchvision.logger.make_grid(x, self.grid_size, padding=0)
        #     self.returnFunc = make_grid
        #

    # def get_tile(self, img, n):
    #     """Return the augmentation tile of picture
    #
    #     :param img:
    #     :param n:
    #     :return:
    #     """
    #
    #     w = float(img.size[0]) / self.grid_size
    #     y = int(n / self.grid_size)
    #     x = n % self.grid_size
    #     tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
    #     tile = self._augment_tile(tile)
    #     return tile

    def _get_image_of_index(self, index):
        framename = self._data_path + '/' + self._data_paths[index]
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img)

    def __getitem__(self, index):
        # TODO(lyj):wait to check
        """

        :param index:
        :return:
        """
        img = self._get_image_of_index(index)
        # n_grids = self.grid_size ** 2
        # tiles = [None] * n_grids
        # for n in range(n_grids):
        #     tiles[n] = self.get_tile(img, n)

        # order = np.random.randint(len(self.permutations) + 1)  # added 1 for class 0: unsorted
        order = np.random.randint(4)

        if self._bias_whole_image:
            if self._bias_whole_image > random():
                order = 0
        if order == 0:
            # data = tiles
            data = self._augment_tile(img)
        else:
            # data = [tiles[self.permutations[order - 1][t]] for t in range(n_grids)]
            data = self._augment_tile(img.transpose(order + 1))
            #

            # TODO(lyj):transform the data according to the order

        # data = torch.stack(data, 0)
        # return self.returnFunc(data), int(order), int(self.labels[index])
        return data, int(order), int(self._labels[index])

    def __len__(self):
        return len(self._data_paths)

    # def __retrieve_permutations(self, classes):
    #     all_perm = np.load('permutations_%d.npy' % (classes))
    #     # from range [1,9] to [0,8]
    #     if all_perm.min() == 1:
    #         all_perm = all_perm - 1
    #
    #     return all_perm


class RotationTestDataset(RotationTrainDataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)

    def __getitem__(self, index):
        # framename = self.data_path + '/' + self.data_paths[index]
        # img = Image.open(framename).convert('RGB')
        img = super(RotationTestDataset, self)._get_image_of_index(index)
        # return self._image_transformer(img), 0, int(self.labels[index])
        return img, 0, int(self._labels[index])


class RotationTestDatasetMultiple(RotationTrainDataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)
        self._image_transformer = transforms.Compose([
            transforms.Resize(255, Image.BILINEAR),
        ])
        self._image_transformer_full = transforms.Compose([
            transforms.Resize(225, Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self._augment_tile = transforms.Compose([
            transforms.Resize((75, 75), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        framename = self._data_path + '/' + self._data_paths[index]
        _img = Image.open(framename).convert('RGB')
        img = self._image_transformer(_img)

        w = float(img.size[0]) / self.grid_size
        n_grids = self.grid_size ** 2
        images = []
        jig_labels = []
        tiles = [None] * n_grids
        for n in range(n_grids):
            y = int(n / self.grid_size)
            x = n % self.grid_size
            tile = img.crop([x * w, y * w, (x + 1) * w, (y + 1) * w])
            tile = self._augment_tile(tile)
            tiles[n] = tile
        for order in range(0, len(self.permutations) + 1, 3):
            if order == 0:
                data = tiles
            else:
                data = [tiles[self.permutations[order - 1][t]] for t in range(n_grids)]
            data = self.returnFunc(torch.stack(data, 0))
            images.append(data)
            jig_labels.append(order)
        images = torch.stack(images, 0)
        jig_labels = torch.LongTensor(jig_labels)
        return images, jig_labels, int(self._labels[index])


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, limit):
        indices = torch.randperm(len(dataset))[:limit]
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


class ConcatDataset(Dataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.

    Arguments:
        datasets (sequence): List of datasets to be concatenated
    """

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    def isMulti(self):
        return isinstance(self.datasets[0], RotationTestDatasetMultiple)

    def __init__(self, datasets):
        super(ConcatDataset, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        self.cumulative_sizes = self.cumsum(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], dataset_idx

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes