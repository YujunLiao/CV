from .utils.ds2dl import train_DL_fn, test_DL_fn
from .utils.get_p_l import get_p_l
from ..dataset.internal_rot import InternalRotTrain, InternalRotTest


def get_DGIR_data_loader(sources='', target='', data_dir='', val_size=float(0),
                        prob=float(0), batch_size=128, _max=-1, args=None):
    train_paths, val_paths, train_labels, val_labels = \
        get_p_l(sources, dir=data_dir + 'train/', val_size=val_size)
    test_paths, _, test_labels, _ = get_p_l(target, dir=data_dir + 'test/')

    # dataset
    train_DS = InternalRotTrain(train_paths, train_labels, prob=prob, _max=_max,
                                args=args, img_size=args.image_size, margin=args.margin)
    val_s_DS = InternalRotTest(val_paths, val_labels, prob=1, _max=_max,
                             args=args, img_size=args.image_size, margin=args.margin)
    val_us_DS = InternalRotTest(val_paths, val_labels, prob=0.25, _max=_max,
                               args=args, img_size=args.image_size, margin=args.margin)

    test_s_DS = InternalRotTest(test_paths, test_labels, prob=1, _max=_max,
                              args=args, img_size=args.image_size, margin=args.margin)
    test_us_DS = InternalRotTest(test_paths, test_labels, prob=0.25, _max=_max,
                                args=args, img_size=args.image_size, margin=args.margin)

    return train_DL_fn(train_DS, batch_size),\
           test_DL_fn(val_s_DS, batch_size), test_DL_fn(val_us_DS, batch_size),\
           test_DL_fn(test_s_DS, batch_size), test_DL_fn(test_us_DS, batch_size)