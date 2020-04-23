from .fns import train_DL_fn, test_DL_fn
from .utils import get_data
from ..dataset.rotation import Rotation

def get_DGR_data_loader(sources='', target='', data_dir='', val_size=float(0),
                        prob=float(0), batch_size=128, _max=-1):
    train_paths, val_paths, train_labels, val_labels = \
        get_data(sources, dir=data_dir+'train/', val_size=val_size)
    test_paths, _, test_labels, _ = get_data(target, dir=data_dir+'test/')

    # dataset
    train_DS = Rotation(train_paths, train_labels, prob=prob, _max=_max)
    val_DS = Rotation(val_paths, val_labels, prob=0, _max=_max)
    test_DS = Rotation(test_paths, test_labels, prob=0, _max=_max)

    return train_DL_fn(train_DS, batch_size), test_DL_fn(val_DS, batch_size),\
           test_DL_fn(test_DS, batch_size)















