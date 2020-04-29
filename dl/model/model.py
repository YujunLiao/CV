from . import caffenet, mnist, resnet

model_fns = {
    'caffenet': caffenet.caffenet,
    'resnet18': resnet.resnet18,
    # 'alexnet': alexnet.alexnet,
    'resnet50': resnet.resnet50,
    'lenet': mnist.lenet
}


def get_model(network, **kwargs):
    model = model_fns[network](
        # Jigsaw class of 0 refers to original picture, apart from the original one, there
        # are another 30 classes, in total 31 classes of jigsaw pictures.
        # jigsaw_classes=training_arguments.jigsaw_n_classes + 1,

        # When using rotation technology as the unsupervised task, there are in total
        # 4 classes, which are original one, 90, 180, 270 degree.
        **kwargs
    )
    return model