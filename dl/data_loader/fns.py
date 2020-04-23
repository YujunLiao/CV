from torch.utils.data import DataLoader


train_DL_fn = lambda DS, bs: DataLoader(
        DS,
        batch_size=bs,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True)

test_DL_fn = lambda DS, bs: DataLoader(
        DS,
        batch_size=bs,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False)