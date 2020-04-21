from torch import optim

class MyOptimizer:
    """Return my optimizer.

    Implementation:
        optimizer

    """
    def __init__(self,model, lr=0.001, train_all=True, nesterov=False):
        if train_all:
            model_parameters = model.parameters()
        else:
            model_parameters = model.get_params(lr)
        self.optimizer = optim.SGD(
            model_parameters,
            weight_decay=.0005,
            momentum=.9,
            nesterov=nesterov,
            lr=lr
        )



def get_optimizer(model, lr=0.001, train_all=True, nesterov=False):
        if train_all:
            model_parameters = model.parameters()
        else:
            model_parameters = model.get_params(lr)
        return optim.SGD(
            model_parameters,
            weight_decay=.0005,
            momentum=.9,
            nesterov=nesterov,
            lr=lr
        )