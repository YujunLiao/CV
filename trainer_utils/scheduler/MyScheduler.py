from torch import optim

class MyScheduler:
    def __init__(self, my_training_arguments, my_optimizer):
        step_size = int(my_training_arguments.training_arguments.epochs * .8)
        self.scheduler = optim.lr_scheduler.StepLR(my_optimizer.optimizer, step_size)