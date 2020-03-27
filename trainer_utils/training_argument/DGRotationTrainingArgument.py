import argparse
from trainer_utils.data_loader.dataset.PDataset import available_domains
from trainer_utils.model.MyModel import model_dictionary


def parameters_lists(arg):
    return [float(x) for x in arg.split(',')]


class DGRotationTrainingArgument(argparse.ArgumentParser):
    """Store the arguments coming from the console.

    Implementation:
        set_arguments_dictionary:
        training_arguments:A dictionary containing all the arguments from the console.
    """
    def __init__(self):
        super(DGRotationTrainingArgument, self).__init__(
            description="Script to launch jigsaw training",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )

        self._get_arguments_from_console()
        self.training_arguments = self.parse_args()

    def _get_arguments_from_console(self):
        """Get the training arguments from the console.

        :return:
        """
        # Arguments for training
        self.add_argument("--network", choices=model_dictionary.keys(), help="Which network to use")
        self.add_argument("--source", choices=available_domains, help="Source", nargs='+')
        self.add_argument("--target", choices=available_domains, help="Target")
        self.add_argument("--n_classes", "-c", type=int, help="Number of classes")
        self.add_argument("--number_of_unsupervised_classes", type=int, default=4, help="Number of classes for the jigsaw task")
        self.add_argument("--unsupervised_task_weight", type=float, help="Weight for the jigsaw puzzle")

        self.add_argument("--ooo_weight", type=float, default=0, help="Weight for odd one out task")
        # image_size: For example, an image's size is 3*225*225
        self.add_argument("--image_size", type=int, default=225, help="Image size")
        # In general, in each epoch, all the samples are used for one time.
        self.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
        # batch_size (int, optional): The number of samples used for training in each iteration.
        self.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size")
        self.add_argument("--val_size", type=float, default="0.1", help="Validation size (between 0 and 1)")
        self.add_argument("--learning_rate", "-l", type=float, default=.01, help="Learning rate")
        self.add_argument("--bias_whole_image", default=None, type=float,
                          help="If set, will bias the training procedure to show more often the whole image")
        self.add_argument("--classify_only_ordered_images_or_not", type=bool,
                          help="If true, the network will only try to classify the non scrambled images")

        # Argument for logger.
        self.add_argument("--tf_logger", type=bool, default=True,
                            help="If true will save tensorboard compatible logs")
        self.add_argument("--folder_name", default=None, help="Used by the logger to save logs")

        #
        self.add_argument("--limit_source", default=None, type=int,
                            help="If set, it will limit the number of training samples")
        self.add_argument("--limit_target", default=None, type=int,
                            help="If set, it will limit the number of testing samples")


        self.add_argument("--train_all", default=True, type=bool,
                            help="If true, all network weights will be trained")
        self.add_argument("--suffix", default="", help="Suffix for the logger")
        self.add_argument("--nesterov", default=False, type=bool, help="Use nesterov")

        # Arguments for Test-Time Augmentation
        self.add_argument("--TTA", type=bool, default=False, help="Activate test time data augmentation")
        self.add_argument("--min_scale", default=0.8, type=float, help="Minimum scale percent")
        self.add_argument("--max_scale", default=1.0, type=float, help="Maximum scale percent")
        self.add_argument("--random_horiz_flip", default=0.0, type=float, help="Chance of random horizontal flip")
        self.add_argument("--jitter", default=0.0, type=float, help="Color jitter amount")
        self.add_argument("--tile_random_grayscale", default=0.1, type=float,
                            help="Chance of randomly greyscaling a tile")


        # Arguments for lazy training
        self.add_argument("--domains_list", nargs='+')
        self.add_argument("--target_domain_list", nargs='+')
        self.add_argument("--parameters_lists", type=parameters_lists, nargs='+')
        # self.add_argument("--parameters_lists", type=tuple, nargs='+')
        self.add_argument("--repeat_times", type=int)
        self.add_argument("--redirect_to_file", default=0, type=int)


