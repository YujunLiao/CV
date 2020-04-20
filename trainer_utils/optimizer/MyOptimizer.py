from torch import optim

class MyOptimizer:
    """Return my optimizer.

    Implementation:
        optimizer

    """
    def __init__(self, my_training_arguments, my_model):
        if my_training_arguments.args.train_all:
            model_parameters = my_model.parameters()
        else:
            model_parameters = my_model.get_params(my_training_arguments.args.learning_rate)
        self.optimizer = optim.SGD(
            model_parameters,
            weight_decay=.0005,
            momentum=.9,
            nesterov=my_training_arguments.args.nesterov,
            lr=my_training_arguments.args.learning_rate
        )