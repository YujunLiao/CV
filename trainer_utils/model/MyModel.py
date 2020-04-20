# from model.pretrained import caffenet, resnet, mnist, alexnet
from trainer_utils.model.pretrained import resnet
from trainer_utils.model.pretrained import caffenet, mnist

model_dictionary = {
    'caffenet': caffenet.get_caffenet,
    'resnet18': resnet.resnet18,
    # 'alexnet': alexnet.alexnet,
    'resnet50': resnet.resnet50,
    'lenet': mnist.lenet
}


class MyModel:
    """Return pre-trained model.
    back bone network.

    Implementation:
        model
    """
    def __init__(self, my_training_arguments):
        """

        :param my_training_arguments:Name of network, the number of classes of unsupervised and
        supervised task.
        """
        self._get_model_function = model_dictionary[my_training_arguments.network]
        self.model = self._get_model_function(
            # Jigsaw class of 0 refers to original picture, apart from the original one, there
            # are another 30 classes, in total 31 classes of jigsaw pictures.
            # jigsaw_classes=training_arguments.jigsaw_n_classes + 1,

            # When using rotation technology as the unsupervised task, there are in total
            # 4 classes, which are original one, 90, 180, 270 degree.
            jigsaw_classes=my_training_arguments.number_of_unsupervised_classes,
            classes=my_training_arguments.n_classes
        )





        # self.model = self._get_model(my_training_arguments.training_arguments.network)(
        #     # Jigsaw class of 0 refers to original picture, apart from the original one, there
        #     # are another 30 classes, in total 31 classes of jigsaw pictures.
        #     # jigsaw_classes=training_arguments.jigsaw_n_classes + 1,
        #
        #     # When using rotation technology as the unsupervised task, there are in total
        #     # 4 classes, which are original one, 90, 180, 270 degree.
        #     jigsaw_classes = 4,
        #     classes=my_training_arguments.training_arguments.n_classes
        # )

    # def _get_model(self, model_name):
    #     """Return the function of getting the network.
    #
    #     :param model_name: Name of the network used for extract features.
    #     :return:{function}
    #     """
    #     if model_name not in model_dictionary:
    #         raise ValueError('Name of network unknown %s' % model_name)
    #
    #     def get_network_fn(**kwargs):
    #         """
    #
    #         :param kwargs:
    #         :return:
    #         """
    #         return model_dictionary[model_name](**kwargs)
    #
    #     return get_network_fn

def get_model(my_training_arguments):
    _get_model_function = model_dictionary[my_training_arguments.network]
    model = _get_model_function(
        # Jigsaw class of 0 refers to original picture, apart from the original one, there
        # are another 30 classes, in total 31 classes of jigsaw pictures.
        # jigsaw_classes=training_arguments.jigsaw_n_classes + 1,

        # When using rotation technology as the unsupervised task, there are in total
        # 4 classes, which are original one, 90, 180, 270 degree.
        jigsaw_classes=my_training_arguments.number_of_unsupervised_classes,
        classes=my_training_arguments.n_classes
    )
    return model





        # self.model = self._get_model(my_training_arguments.training_arguments.network)(
        #     # Jigsaw class of 0 refers to original picture, apart from the original one, there
        #     # are another 30 classes, in total 31 classes of jigsaw pictures.
        #     # jigsaw_classes=training_arguments.jigsaw_n_classes + 1,
        #
        #     # When using rotation technology as the unsupervised task, there are in total
        #     # 4 classes, which are original one, 90, 180, 270 degree.
        #     jigsaw_classes = 4,
        #     classes=my_training_arguments.training_arguments.n_classes
        # )

    # def _get_model(self, model_name):
    #     """Return the function of getting the network.
    #
    #     :param model_name: Name of the network used for extract features.
    #     :return:{function}
    #     """
    #     if model_name not in model_dictionary:
    #         raise ValueError('Name of network unknown %s' % model_name)
    #
    #     def get_network_fn(**kwargs):
    #         """
    #
    #         :param kwargs:
    #         :return:
    #         """
    #         return model_dictionary[model_name](**kwargs)
    #
    #     return get_network_fn