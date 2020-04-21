
import sys
from tensorboardX import SummaryWriter
from torch import optim
import os


import torch
from torch import nn
from torch.nn import functional as F
from time import time, strftime, localtime
import argparse
# from trainer_utils.model.MyModel import MyModel
from trainer_utils.model.MyModel import MyModel, get_model


from trainer_utils.logger.Logger import Logger
from trainer_utils.data_loader.DGRotationDataLoader import DGRotationDataLoader
from trainer_utils.optimizer.MyOptimizer import MyOptimizer, get_optimizer
from trainer_utils.scheduler.MyScheduler import MyScheduler
from trainer_utils.output_manager.OutputManager import OutputManager
from trainer_utils.lazy_man.LazyMan import LazyMan, LazyMan2
import socket


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network")
    parser.add_argument("--source", nargs='+')
    parser.add_argument("--target", help="Target")
    parser.add_argument("--n_classes", "-c", type=int)
    parser.add_argument("--number_of_unsupervised_classes", type=int, default=4)
    parser.add_argument("--unsupervised_task_weight", type=float)

    parser.add_argument("--ooo_weight", type=float, default=0)
    # image_size: For example, an image's size is 3*225*225
    parser.add_argument("--image_size", type=int, default=225)
    # In general, in each epoch, all the samples are used for one time.
    parser.add_argument("--epochs", "-e", type=int, default=30)
    # batch_size (int, optional): The number of samples used for training in each iteration.
    parser.add_argument("--batch_size", "-b", type=int, default=128)
    parser.add_argument("--val_size", type=float, default="0.1")
    parser.add_argument("--learning_rate", "-l", type=float, default=.01)
    parser.add_argument("--bias_whole_image", default=None, type=float)
    parser.add_argument("--classify_only_ordered_images_or_not", type=bool)

    # Argument for logger.
    parser.add_argument("--tf_logger", type=bool, default=True)
    parser.add_argument("--folder_name", default=None)
    parser.add_argument("--log_dir",
                        default='/home/giorgio/Files/pycharm_project/CV/trainer_utils/output_manager/output_file/')
    parser.add_argument("--log_name", default='log')
    parser.add_argument("--experiment", default='DG_rot')

    #
    parser.add_argument("--limit_source", default=None, type=int)
    parser.add_argument("--limit_target", default=None, type=int)

    parser.add_argument("--train_all", default=True, type=bool)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--nesterov", default=False, type=bool)

    # Arguments for Test-Time Augmentation
    parser.add_argument("--TTA", type=bool, default=False)
    parser.add_argument("--min_scale", default=0.8, type=float)
    parser.add_argument("--max_scale", default=1.0, type=float)
    parser.add_argument("--random_horiz_flip", default=0.0, type=float)
    parser.add_argument("--jitter", default=0.0, type=float)
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float)

    # Arguments for lazy training
    parser.add_argument("--domains_list", nargs='+')
    parser.add_argument("--target_domain_list", nargs='+')
    parser.add_argument("--parameters_lists", nargs='+'
                        , type=lambda params:[float(_) for _ in params.split(',')])
    # parser.add_argument("--parameters_lists", type=tuple, nargs='+')
    parser.add_argument("--repeat_times", type=int)
    parser.add_argument("--redirect_to_file", default=0, type=int)

    parser.add_argument(
        '--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    return parser.parse_args()


class Trainer:
    def __init__(self, args, model, data_loader, optimizer, scheduler, output_manager):
        # self.args = args.args
        self.args = args
        self.device = args.device
        self.model = model.to(args.device)
        self.output_manager=output_manager

        self.train_data_loader = data_loader.train_data_loader
        self.validation_data_loader = data_loader.validation_data_loader
        self.test_data_loader = data_loader.test_data_loader

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.classify_only_ordered_images_or_not = self.args.classify_only_ordered_images_or_not
        self.number_of_images_classes = self.args.n_classes
        self.test_loaders = {"val": self.validation_data_loader, "test": self.test_data_loader}

        self.cur_epoch = -1
        self.log_freq = 30

        self.do_training()


    def _do_epoch(self):
        self.cur_epoch += 1
        self.scheduler.step()
        lrs = self.scheduler.get_lr()
        p = OutputManager.print
        criterion = nn.CrossEntropyLoss()

        # Set the mode of the model to trainer, then the parameters can begin to be trained
        self.model.train()
        # domain_index_of_images_in_this_patch is target domain index in the source domain list
        for i, ((data, rotation_label, class_label), domain_index_of_images_in_this_patch) in enumerate(self.train_data_loader):
            data, rotation_label, class_label, domain_index_of_images_in_this_patch = data.to(self.device), rotation_label.to(self.device), class_label.to(self.device), domain_index_of_images_in_this_patch.to(self.device)
            self.optimizer.zero_grad()

            rotation_predict_label, class_predict_label = self.model(data)  # , lambda_val=lambda_val)
            unsupervised_task_loss = criterion(rotation_predict_label, rotation_label)
            supervised_task_loss = criterion(class_predict_label[rotation_label == 0], class_label[rotation_label == 0])

            _, cls_pred = class_predict_label.max(dim=1)
            _, jig_pred = rotation_predict_label.max(dim=1)
            # _, domain_pred = domain_logit.max(dim=1)
            loss = supervised_task_loss + unsupervised_task_loss * self.args.unsupervised_task_weight

            loss.backward()
            self.optimizer.step()
            if i%self.log_freq == 0:
                p([f'epoch:{self.cur_epoch}/{self.args.epochs};lr:{" ".join([str(lr) for lr in lrs])};'+\
                   f'bs:{data.shape[0]};{i}/{len(self.train_data_loader)}',
                   f'train_acc:j:{torch.sum(jig_pred == rotation_label.data).item()/data.shape[0]};'+\
                      f'c:{torch.sum(cls_pred == class_label.data).item()/data.shape[0]}',
                   f'train_loss:j:{unsupervised_task_loss.item()};c:{supervised_task_loss.item()}'])

            del loss, supervised_task_loss, unsupervised_task_loss, rotation_predict_label, class_predict_label

        # eval
        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)
                class_correct = Trainer.eval(self.model, loader, device=self.device)
                class_acc = float(class_correct) / total
                p(f'{phase}_acc:c:{class_acc}')
                self.results[phase][self.current_epoch] = class_acc

    @staticmethod
    def eval(model, loader, device='cpu'):
        class_correct = 0
        for _, ((data, _, class_l), _) in enumerate(loader):
            data, _, class_l = data.to(device), _.to(device), class_l.to(device)
            _, class_logit = model(data)
            _, cls_pred = class_logit.max(dim=1)
            class_correct += torch.sum(cls_pred == class_l.data)
        return class_correct


    def do_training(self):
        start_time = time()
        p = OutputManager.print
        p('Start training')
        p({
            'train':self.train_data_loader.dataset,
            'validation': self.validation_data_loader.dataset,
            'test': self.test_data_loader.dataset,
        })
        p(vars(self.args))

        # TODO(lyj):
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        for self.current_epoch in range(self.args.epochs):
            self._do_epoch()
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()

        # print("Best val %g, corresponding test %g - best test: %g" % (val_res.max(), test_res[idx_best], test_res.max()))
        p(strftime("%Y-%m-%d %H:%M:%S", localtime()))
        temp_dict = {
            'source': self.args.source,
            'target':self.args.target,
            'param':self.args.parameters_lists,
            'Highest accuracy on validation set appears on epoch': val_res.argmax().data,
            'Highest accuracy on test set appears on epoch ': test_res.argmax().data,
            'Accuracy on test set when the accuracy on validation set is highest:%.3f': test_res[idx_best],
            'Highest accuracy on test set:%.3f': test_res.max(),

        }
        p(temp_dict)



        self.output_manager.write_to_output_file([
            '--------------------------------------------------------',
            str(strftime("%Y-%m-%d %H:%M:%S", localtime())),
            self.args.source,
            "target domain:" + self.args.target,
            "jigweight:" + str(self.args.unsupervised_task_weight),
            "bias_hole_image:" + str(self.args.bias_whole_image),
            "only_classify the ordered image:" + str(self.args.classify_only_ordered_images_or_not),
            "batch_size:" + str(self.args.batch_size) + " learning_rate:" + str(self.args.learning_rate),
            "Highest accuracy on validation set appears on epoch " + str(val_res.argmax().data),
            "Highest accuracy on test set appears on epoch " + str(test_res.argmax().data),
            str("Accuracy on test set when the accuracy on validation set is highest:%.3f" % test_res[idx_best]),
            str("Highest accuracy on test set:%.3f" % test_res.max()),
            str("It took %g" % (time() - start_time))
        ])

class BaseTrainer:
    def __init__(self, args, model, data_loader, optimizer, scheduler, output_manager):
        # self.args = args.args
        self.args = args
        self.device = args.device
        self.model = model.to(args.device)
        self.output_manager=output_manager

        self.train_data_loader = data_loader.train_data_loader
        self.validation_data_loader = data_loader.validation_data_loader
        self.test_data_loader = data_loader.test_data_loader

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.classify_only_ordered_images_or_not = self.args.classify_only_ordered_images_or_not
        self.number_of_images_classes = self.args.n_classes
        self.test_loaders = {"val": self.validation_data_loader, "test": self.test_data_loader}

        self.cur_epoch = -1
        self.log_freq = 30

        self.do_training()


    def _do_epoch(self):
        self.cur_epoch += 1
        self.scheduler.step()
        lrs = self.scheduler.get_lr()
        p = OutputManager.print
        criterion = nn.CrossEntropyLoss()

        # Set the mode of the model to trainer, then the parameters can begin to be trained
        self.model.train()
        # domain_index_of_images_in_this_patch is target domain index in the source domain list
        for i, ((data, rotation_label, class_label), domain_index_of_images_in_this_patch) in enumerate(self.train_data_loader):
            data, rotation_label, class_label, domain_index_of_images_in_this_patch = data.to(self.device), rotation_label.to(self.device), class_label.to(self.device), domain_index_of_images_in_this_patch.to(self.device)
            self.optimizer.zero_grad()

            rotation_predict_label, class_predict_label = self.model(data)  # , lambda_val=lambda_val)
            unsupervised_task_loss = criterion(rotation_predict_label, rotation_label)
            supervised_task_loss = criterion(class_predict_label[rotation_label == 0], class_label[rotation_label == 0])

            _, cls_pred = class_predict_label.max(dim=1)
            _, jig_pred = rotation_predict_label.max(dim=1)
            # _, domain_pred = domain_logit.max(dim=1)
            loss = supervised_task_loss + unsupervised_task_loss * self.args.unsupervised_task_weight

            loss.backward()
            self.optimizer.step()
            if i%self.log_freq == 0:
                p([f'epoch:{self.cur_epoch}/{self.args.epochs};lr:{" ".join([str(lr) for lr in lrs])};'+\
                   f'bs:{data.shape[0]};{i}/{len(self.train_data_loader)}',
                   f'train_acc:j:{torch.sum(jig_pred == rotation_label.data).item()/data.shape[0]};'+\
                      f'c:{torch.sum(cls_pred == class_label.data).item()/data.shape[0]}',
                   f'train_loss:j:{unsupervised_task_loss.item()};c:{supervised_task_loss.item()}'])

            del loss, supervised_task_loss, unsupervised_task_loss, rotation_predict_label, class_predict_label

        # eval
        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)
                class_correct = Trainer.eval(self.model, loader, device=self.device)
                class_acc = float(class_correct) / total
                p(f'{phase}_acc:c:{class_acc}')
                self.results[phase][self.current_epoch] = class_acc

    @staticmethod
    def eval(model, loader, device='cpu'):
        class_correct = 0
        for _, ((data, _, class_l), _) in enumerate(loader):
            data, _, class_l = data.to(device), _.to(device), class_l.to(device)
            _, class_logit = model(data)
            _, cls_pred = class_logit.max(dim=1)
            class_correct += torch.sum(cls_pred == class_l.data)
        return class_correct


    def do_training(self):
        start_time = time()
        p = OutputManager.print
        p('Start training')
        p({
            'train':self.train_data_loader.dataset,
            'validation': self.validation_data_loader.dataset,
            'test': self.test_data_loader.dataset,
        })
        p(vars(self.args))

        # TODO(lyj):
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        for self.current_epoch in range(self.args.epochs):
            self._do_epoch()
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()

        # print("Best val %g, corresponding test %g - best test: %g" % (val_res.max(), test_res[idx_best], test_res.max()))
        p(strftime("%Y-%m-%d %H:%M:%S", localtime()))
        temp_dict = {
            'source': self.args.source,
            'target':self.args.target,
            'param':self.args.parameters_lists,
            'Highest accuracy on validation set appears on epoch': val_res.argmax().data,
            'Highest accuracy on test set appears on epoch ': test_res.argmax().data,
            'Accuracy on test set when the accuracy on validation set is highest:%.3f': test_res[idx_best],
            'Highest accuracy on test set:%.3f': test_res.max(),

        }
        p(temp_dict)



        self.output_manager.write_to_output_file([
            '--------------------------------------------------------',
            str(strftime("%Y-%m-%d %H:%M:%S", localtime())),
            self.args.source,
            "target domain:" + self.args.target,
            "jigweight:" + str(self.args.unsupervised_task_weight),
            "bias_hole_image:" + str(self.args.bias_whole_image),
            "only_classify the ordered image:" + str(self.args.classify_only_ordered_images_or_not),
            "batch_size:" + str(self.args.batch_size) + " learning_rate:" + str(self.args.learning_rate),
            "Highest accuracy on validation set appears on epoch " + str(val_res.argmax().data),
            "Highest accuracy on test set appears on epoch " + str(test_res.argmax().data),
            str("Accuracy on test set when the accuracy on validation set is highest:%.3f" % test_res[idx_best]),
            str("Highest accuracy on test set:%.3f" % test_res.max()),
            str("It took %g" % (time() - start_time))
        ])

class DGTrainer(BaseTrainer):





class Container:
    def __init__(self):
        pass

def lazy_train(args, output_manager):
    model = get_model(args.network,
                      jigsaw_classes=args.number_of_unsupervised_classes,
                      classes=args.n_classes)
    is_patch_based_or_not = model.is_patch_based()
    temp = Container()
    temp.training_arguments = args
    temp.args = args
    data_loader = DGRotationDataLoader(temp, is_patch_based_or_not)
    optimizer = get_optimizer(model, lr=args.learning_rate, train_all=args.train_all)
    scheduler = optim.lr_scheduler.StepLR(optimizer, int(args.epochs * .8))
    # scheduler = MyScheduler(temp, optimizer)
    Trainer(args, model, data_loader, optimizer, scheduler, output_manager)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args = get_args()
    # my_training_arguments.training_arguments.classify_only_ordered_images_or_not=True
    args.TTA = False
    args.nesterov = False

    for parameter_pair in args.parameters_lists:

        args.unsupervised_task_weight=parameter_pair[0]
        args.bias_whole_image=parameter_pair[1]
        # lazy_man = LazyMan(['CALTECH', 'LABELME', 'PASCAL', 'SUN'])
        # lazy_man = LazyMan(
        #     ['art_painting', 'cartoon', 'sketch', 'photo'],
        #     ['art_painting', 'cartoon', 'sketch', 'photo']
        # )
        lazy_man = LazyMan2(
            args.domains_list,
            args.target_domain_list
        )

        output_file_path = \
            f'{args.log_dir}/{socket.gethostname()}/{args.experiment}/{args.network}/'+\
            f'{str(args.unsupervised_task_weight)}_{str(args.bias_whole_image)}/'

        if args.redirect_to_file == 1:
            if not os.path.exists(output_file_path):
                os.makedirs(output_file_path)
            orig_stdout = sys.stdout
            f = open(output_file_path+args.log_name, 'w')
            sys.stdout = f

        for source_and_target_domain in lazy_man.source_and_target_domain_permutation_list:
            args.source=source_and_target_domain['source_domain']
            args.target=source_and_target_domain['target_domain']
            #
            output_manager = OutputManager(
                output_file_path=output_file_path,
                output_file_name=args.source[0] + '_' + args.target

            )
            # output_manager = Container()
            # writer = SummaryWriter(log_dir=args.log_dir)
            for i in range(int(args.repeat_times)):
                lazy_train(args, output_manager)



