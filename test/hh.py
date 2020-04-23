from dl.dataset.get_data import get_data
from dl.dataset import BaseDataset

import torchvision.transforms as transforms

tfs = [transforms.Resize((128, 128)), transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

paths, _, labels, _ = get_data('cartoon')
ds = BaseDataset(paths, labels)
# t_ds = torch.tensor(ds)

#img, n = Rotation.rotate(, prob=0.1)
img_1 = transforms.Compose(tfs)(ds[0][0])
img_2 = transforms.ToTensor()(ds[0][0])


print()