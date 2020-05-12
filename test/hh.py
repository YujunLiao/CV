from dl.data_loader.utils.get_p_l import get_p_l
from dl.dataset.tf_fn import norm_tf_fn

import torchvision.transforms as transforms

tfs = [transforms.Resize((128, 128)), transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

paths, _, labels, _ = get_p_l('cartoon')
ds = BaseDataset(paths, labels)
# t_ds = torch.tensor(ds)

#img, n = Rotation.rotate(, prob=0.1)
img_1 = transforms.Compose(tfs)(ds[0][0])
img_2 = transforms.ToTensor()(ds[0][0])


print()