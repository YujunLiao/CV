import torchvision.transforms as tf

def train_tf_fn(args):
    tf_fns = []
    tf_fns.append(tf.RandomResizedCrop(
        (args.image_size, args.image_size),
        (args.min_scale, args.max_scale),
    ))
    if args.random_horiz_flip > 0.0:
        tf_fns.append(tf.RandomHorizontalFlip(args.random_horiz_flip))
    if args.jitter > 0.0:
        tf_fns.append(tf.ColorJitter(
            brightness=args.jitter,
            contrast=args.jitter,
            saturation=args.jitter,
            hue=min(0.5, args.jitter),
        ))
    return tf.Compose(tf_fns)


def test_tf_fn(args):
    tf_fns = []
    tf_fns.append(tf.Resize((args.image_size, args.image_size)))
    return tf.Compose(tf_fns)


def tile_tf_fn(args):
    tf_fns = []
    if args.tile_random_grayscale:
        tf_fns.append(tf.RandomGrayscale(args.tile_random_grayscale))
    return tf.Compose(tf_fns)

norm_tf_fn = tf.Compose([
    tf.ToTensor(),
    tf.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

