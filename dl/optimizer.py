from torch import optim


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