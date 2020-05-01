from PIL import Image
import torch.utils.data as data


class BaseDataset(data.Dataset):
    """:return A PIL image with specified size

    """
    def __init__(self, paths, labels, _max=-1):
        super().__init__()
        if _max != -1 and len(paths) > _max:
            paths = paths[:_max]
            labels = labels[:_max]
        self.dataset_length = len(paths)
        self.paths = paths
        self.labels = labels

    def __getitem__(self, index):
        return Image.open(self.paths[index]).convert('RGB'), \
               self.labels[index]

    def __len__(self):
        return self.dataset_length

