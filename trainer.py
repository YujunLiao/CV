
import sys
from tensorboardX import SummaryWriter

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
from trainer_utils.optimizer.MyOptimizer import MyOptimizer
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
    def __init__(self, args, my_model, my_data_loader, my_optimizer, my_scheduler, output_manager, ):
        # self.args = args.args
        self.args = args
        self.device = args.device
        self.model = my_model.to(args.device)
        self.output_manager=output_manager

        self.train_data_loader = my_data_loader.train_data_loader
        self.validation_data_loader = my_data_loader.validation_data_loader
        self.test_data_loader = my_data_loader.test_data_loader

        self.optimizer = my_optimizer.optimizer
        self.scheduler = my_scheduler.scheduler

        self.classify_only_ordered_images_or_not = self.args.classify_only_ordered_images_or_not
        self.number_of_images_classes = self.args.n_classes
        self.test_loaders = {"val": self.validation_data_loader, "test": self.test_data_loader}

        if self.args.target in self.args.source:
            self.target_domain_index = self.args.source.index(self.args.target)
            print("Target in source: %d" % self.target_domain_index)
            print(self.args.source)
        else:
            self.target_domain_index = None

    def _do_epoch(self):
        criterion = nn.CrossEntropyLoss()
        # Set the mode of the model to trainer, then the parameters can begin to be trained
        self.model.train()

        # domain_index_of_images_in_this_patch is target domain index in the source domain list
        for i, ((data, rotation_label, class_label), domain_index_of_images_in_this_patch) in enumerate(self.train_data_loader):
            data, rotation_label, class_label, domain_index_of_images_in_this_patch = data.to(self.device), rotation_label.to(self.device), class_label.to(self.device), domain_index_of_images_in_this_patch.to(self.device)
            self.optimizer.zero_grad()

            rotation_predict_label, class_predict_label = self.model(data)  # , lambda_val=lambda_val)
            unsupervised_task_loss = criterion(rotation_predict_label, rotation_label)

            if self.args.classify_only_ordered_images_or_not:
                if self.target_domain_index is not None:
                    # images_should_be_selected_or_not is a 128*1 list containing True or False.
                    images_should_be_selected_or_not = (rotation_label == 0) & (domain_index_of_images_in_this_patch != self.target_domain_index)
                    supervised_task_loss = criterion(
                        class_predict_label[images_should_be_selected_or_not],
                        class_label[images_should_be_selected_or_not]
                    )
                else:
                    supervised_task_loss = criterion(class_predict_label[rotation_label == 0], class_label[rotation_label == 0])

            elif self.target_domain_index:
                supervised_task_loss = criterion(class_predict_label[domain_index_of_images_in_this_patch != self.target_domain_index], class_label[domain_index_of_images_in_this_patch != self.target_domain_index])
            else:
                supervised_task_loss = criterion(class_predict_label, class_label)
            _, cls_pred = class_predict_label.max(dim=1)
            _, jig_pred = rotation_predict_label.max(dim=1)
            # _, domain_pred = domain_logit.max(dim=1)
            loss = supervised_task_loss + unsupervised_task_loss * self.args.unsupervised_task_weight

            loss.backward()
            self.optimizer.step()

            self.logger.log(
                i,
                len(self.train_data_loader),
                {
                    "jigsaw": unsupervised_task_loss.item(),
                    "class": supervised_task_loss.item()
                 },
                {
                    "jigsaw": torch.sum(jig_pred == rotation_label.data).item(),
                    "class": torch.sum(cls_pred == class_label.data).item(),
                 },
                data.shape[0]
            )
            del loss, supervised_task_loss, unsupervised_task_loss, rotation_predict_label, class_predict_label
        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                total = len(loader.dataset)
                if loader.dataset.isMulti():
                    jigsaw_correct, class_correct, single_acc = self.do_test_multi(loader)
                    print("Single vs multi: %g %g" % (float(single_acc) / total, float(class_correct) / total))
                else:
                    jigsaw_correct, class_correct = self.do_test(loader)
                jigsaw_acc = float(jigsaw_correct) / total
                class_acc = float(class_correct) / total
                self.logger.log_test(phase, {"jigsaw": jigsaw_acc, "class": class_acc})
                self.results[phase][self.current_epoch] = class_acc


    def do_test(self, loader):
        jigsaw_correct = 0
        class_correct = 0
        domain_correct = 0
        for it, ((data, jig_l, class_l), _) in enumerate(loader):
            data, jig_l, class_l = data.to(self.device), jig_l.to(self.device), class_l.to(self.device)
            jigsaw_logit, class_logit = self.model(data)
            _, cls_pred = class_logit.max(dim=1)
            _, jig_pred = jigsaw_logit.max(dim=1)
            class_correct += torch.sum(cls_pred == class_l.data)
            jigsaw_correct += torch.sum(jig_pred == jig_l.data)
        return jigsaw_correct, class_correct


    def do_test_multi(self, loader):
        jigsaw_correct = 0
        class_correct = 0
        single_correct = 0
        for it, ((data, jig_l, class_l), d_idx) in enumerate(loader):
            data, jig_l, class_l = data.to(self.device), jig_l.to(self.device), class_l.to(self.device)
            n_permutations = data.shape[1]
            class_logits = torch.zeros(n_permutations, data.shape[0], self.number_of_images_classes).to(self.device)
            for k in range(n_permutations):
                class_logits[k] = F.softmax(self.model(data[:, k])[1], dim=1)
            class_logits[0] *= 4 * n_permutations  # bias more the original image
            class_logit = class_logits.mean(0)
            _, cls_pred = class_logit.max(dim=1)
            jigsaw_logit, single_logit = self.model(data[:, 0])
            _, jig_pred = jigsaw_logit.max(dim=1)
            _, single_logit = single_logit.max(dim=1)
            single_correct += torch.sum(single_logit == class_l.data)
            class_correct += torch.sum(cls_pred == class_l.data)
            jigsaw_correct += torch.sum(jig_pred == jig_l.data[:, 0])
        return jigsaw_correct, class_correct, single_correct

    def do_training(self):
        print('******************Start training**************************************')
        print('--------------------------------------------------------')
        print("Dataset size: trainer %d, val %d, test %d" % (len(self.train_data_loader.dataset), len(self.validation_data_loader.dataset), len(self.test_data_loader.dataset)))
        print('--------------------------------------------------------')
        print(self.args)
        print('--------------------------------------------------------')

        # TODO(lyj):
        self.logger = Logger(self.args, update_frequency=30)  # , "domain", "lambda"
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}
        for self.current_epoch in range(self.args.epochs):
            self.scheduler.step()
            lrs = self.scheduler.get_lr()
            self.logger.new_epoch(lrs)
            print('--------------------------------------------------------')
            print("current epoch:%d", self.current_epoch)
            print("New epoch - lr: %s" % ", ".join([str(lr) for lr in lrs]))
            self._do_epoch()
        val_res = self.results["val"]
        test_res = self.results["test"]
        idx_best = val_res.argmax()

        # print("Best val %g, corresponding test %g - best test: %g" % (val_res.max(), test_res[idx_best], test_res.max()))
        print(strftime("%Y-%m-%d %H:%M:%S", localtime()))
        print(self.args.target)
        print(self.args.source)
        print("unsupervised_task_weight:", self.args.unsupervised_task_weight)
        # TODO(change bias whole image)
        print("bias_hole_image:", self.args.bias_whole_image)
        print("only_classify the ordered image:", self.args.classify_only_ordered_images_or_not)
        print("Highest accuracy on validation set appears on epoch ", val_res.argmax().data)
        print("Highest accuracy on test set appears on epoch ", test_res.argmax().data)
        print("Accuracy on test set when the accuracy on validation set is highest:%.3f" % test_res[idx_best])
        print("Highest accuracy on test set:%.3f" % test_res.max())
        self.logger.save_best(test_res[idx_best], test_res.max())

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
            str("It took %g" % (time() - self.logger.start_time))
        ])

        return self.logger, self.model

class Container:
    def __init__(self):
        pass

def lazy_train(my_training_arguments, output_manager):
    model = get_model(my_training_arguments)
    is_patch_based_or_not = model.is_patch_based()
    temp = Container()
    temp.training_arguments = my_training_arguments
    temp.args = my_training_arguments
    data_loader = DGRotationDataLoader(temp, is_patch_based_or_not)
    optimizer = MyOptimizer(temp, model)
    scheduler = MyScheduler(temp, optimizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = Trainer(temp.training_arguments, model, data_loader, optimizer, scheduler, output_manager)
    trainer.do_training()


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

            output_manager = OutputManager(
                output_file_path=output_file_path,
                output_file_name=args.source[0] + '_' + args.target

            )
            # output_manager = Container()
            # writer = SummaryWriter(log_dir=args.log_dir)
            for i in range(int(args.repeat_times)):
                lazy_train(args, output_manager)



