from PIL import Image
import numpy as np
from .base import BaseDataset
import torchvision.transforms as transforms

class Rotation(BaseDataset):
    def __init__(self, paths, labels, prob=float(0), _max=-1):
        super().__init__(paths, labels, _max)
        self.prob = prob


    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        img, n = Rotation.rotate(img, prob=self.prob)

        tfs = transforms.Compose([transforms.Resize((222, 222)),transforms.ToTensor()])
        return tfs(img), n, label

    @staticmethod
    def rotate(img, prob=float(0)):
        p_ = (1-prob)/3
        n = np.random.choice(np.arange(4), p=[prob, p_, p_, p_])
        if n != 0:
            img = img.transpose(n+1)
        return img, n


