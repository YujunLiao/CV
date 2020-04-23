from PIL import Image
import torch.utils.data as data


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

