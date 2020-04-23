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


class BaseDataset(data.Dataset):
    def __init__(self, paths, labels):
        super().__init__()
        self.dataset_length = len(paths)
        self.paths = paths
        self.labels = labels

    def __getitem__(self, index):
        return Image.open(self.paths[index]).convert('RGB'), \
               self.labels[index]

    def __len__(self):
        return self.dataset_length

