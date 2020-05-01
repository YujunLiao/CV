import copy
import os
import sys
from os.path import dirname
from time import time, strftime, localtime
import socket
from math import ceil
import torch
from torch import optim
from torch import nn
import argparse
import wandb
from dl.data_loader.dgr import get_DGR_data_loader
from dl.data_loader.dgir import get_DGIR_data_loader
from dl.optimizer import get_optimizer
from dl.utils.writer import Writer
from dl.utils.s2t import ms2st, ss2st
from dl.utils.pp import pretty_print as pp
from dl.model import caffenet, resnet, mnist

model_fns = {
    'caffenet': caffenet.caffenet,
    'resnet18': resnet.resnet18,
    # 'alexnet': alexnet.alexnet,
    'resnet50': resnet.resnet50,
    'lenet': mnist.lenet
}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--network", default='resnet18')
    parser.add_argument("--source", nargs='+')
    parser.add_argument("--target")
    parser.add_argument("--num_classes", "-c", type=int, default=7)
    parser.add_argument("--num_usv_classes", type=int, default=4)

    parser.add_argument("--domains", nargs='+',
                        default=['art_painting', 'cartoon', 'sketch', 'photo'])
    parser.add_argument("--targets", nargs='+', default=['art_painting'])
    parser.add_argument("--repeat_times", type=int, default=1)
    parser.add_argument("--parameters", nargs='+', default=[[0.5, 0.5]],
                        type=lambda params:[float(_) for _ in params.split(',')])


    parser.add_argument("--usvt_weight", type=float)
    parser.add_argument("--original_img_prob", default=None, type=float)
    parser.add_argument("--epochs", "-e", type=int, default=2)
    parser.add_argument("--batch_size", "-b", type=int, default=128)
    parser.add_argument("--learning_rate", "-l", type=float, default=.01)
    parser.add_argument("--image_size", type=int, default=222)
    parser.add_argument("--val_size", type=float, default="0.1")
    parser.add_argument("--collect_per_batch", type=int, default=5)
    parser.add_argument("--margin", type=int, default=20)

    # parser.add_argument("--tf_logger", type=bool, default=True)
    # parser.add_argument("--folder_name", default=None)
    parser.add_argument("--data_dir",
                        default=f'{dirname(__file__)}/data/')
    parser.add_argument("--output_dir", default=f'{dirname(__file__)}/output/')
    parser.add_argument("--redirect_to_file", default='null')
    parser.add_argument("--experiment", default='DG_irot')
    parser.add_argument("--wandb", default=False, action='store_true')

    parser.add_argument("--classify_only_original_img", type=bool, default=True)
    parser.add_argument("--max_num_s_img", default=500, type=int)
    parser.add_argument("--max_num_t_img", default=-1, type=int)
    parser.add_argument("--train_all_param", default=True, type=bool)
    parser.add_argument("--nesterov", default=False, action='store_true')

    parser.add_argument("--TTA", default=False, action='store_true')
    parser.add_argument("--min_scale", default=0.8, type=float)
    parser.add_argument("--max_scale", default=1.0, type=float)
    parser.add_argument("--random_horiz_flip", default=0.0, type=float)
    parser.add_argument("--jitter", default=0.0, type=float)
    parser.add_argument("--tile_random_grayscale", default=0.1, type=float)

    return parser.parse_args()


class Trainer:
    def __init__(self, args, model, data_loaders, optimizer, scheduler, writer):
        # self.args = args.args
        self.args = args
        self.device = args.device
        self.model = model.to(args.device)
        self.writer = writer
        self.train_data_loader, self.val_data_loader, self.test_data_loader= data_loaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.test_loaders = {"val": self.val_data_loader, "test": self.test_data_loader}

        self.cur_epoch = -1
        # collect_frequency
        self.collect_per_batch = self.args.collect_per_batch
        self.results = {"val": torch.zeros(self.args.epochs), "test": torch.zeros(self.args.epochs)}

        self.train()

    def train(self):
        start_time = time()
        pp('Start training')
        pp({'train':self.train_data_loader.dataset,
            'validation': self.val_data_loader.dataset,
            'test': self.test_data_loader.dataset})
        pp(vars(self.args))

        # TODO(lyj):
        for self.cur_epoch in range(self.args.epochs):
            self.train_epoch()

        v_b_i = self.results["val"].argmax()
        t_b_i = self.results["test"].argmax()
        test_best = self.results["test"].max()
        test_select = self.results["test"][v_b_i]

        # print("Best val %g, corresponding test %g - best test: %g" % (val_res.max(), test_res[idx_best], test_res.max()))
        temp_dict = {
            'now': strftime("%Y-%m-%d %H:%M:%S", localtime()),
            'source': self.args.source,
            'target':self.args.target,
            'param':self.args.parameters,
            'bs': self.args.batch_size,
            'lr': self.args.learning_rate,
            'Highest accuracy on validation set appears on epoch': t_b_i.item(),
            'Highest accuracy on test set appears on epoch': v_b_i.item(),
            'Accuracy on test set when the accuracy on validation set is highest': test_select.item(),
            'Highest accuracy on test set': test_best.item(),
            'duration': time() - start_time
        }
        pp(temp_dict)
        self.writer.w(temp_dict)

    def train_epoch(self):
        self.scheduler.step()
        lrs = self.scheduler.get_lr()
        criterion = nn.CrossEntropyLoss()

        # Set the mode of the model to trainer, then the parameters can begin to be trained
        self.model.train()
        for i, (data, n, c_l) in enumerate(self.train_data_loader):
            data, n, c_l = data.to(self.device), n.to(self.device), c_l.to(self.device)
            self.optimizer.zero_grad()

            n_logit, c_l_logit = self.model(data)  # , lambda_val=lambda_val)
            usv_loss = criterion(n_logit, n)
            sv_loss = criterion(c_l_logit[n == 0], c_l[n == 0])

            _, c_l_pred = c_l_logit.max(dim=1)
            _, usv_pred = n_logit.max(dim=1)
            # _, domain_pred = domain_logit.max(dim=1)
            loss = sv_loss + usv_loss * self.args.usvt_weight

            loss.backward()
            self.optimizer.step()

            # record and print
            acc_class = torch.sum(c_l_pred == c_l).item() / data.shape[0]
            acc_u = torch.sum(usv_pred == n).item() / data.shape[0]
            if i == 0:
                col_n = ceil(len(self.train_data_loader) / self.collect_per_batch)
                print(f'epoch:{self.cur_epoch}/{self.args.epochs};bs:{data.shape[0]};'
                      f'lr:{" ".join([str(lr) for lr in lrs])}; '
                      f'{len(self.train_data_loader)}/{self.collect_per_batch}={col_n}|', end='')
            if self.args.wandb and self.args.nth_repeat == 0 and i % self.collect_per_batch == 0:
                print('#', end='')
                wandb.log({'acc/train/sv_task': acc_class,
                            'acc/train/usv_task': acc_u,
                            'loss/train/class': sv_loss.item(),
                            'loss/train/usv_task': usv_loss.item(),
                            'loss/train/sum': loss.item()})

            if i == len(self.train_data_loader) - 1:
                print()
                pp([f'train_acc:u:{acc_class};c:{acc_u}'
                    f'train_loss:j:{usv_loss.item()};c:{sv_loss.item()}'])

            del loss, sv_loss, usv_loss, n_logit, c_l_logit

        # eval
        self.model.eval()
        with torch.no_grad():
            for phase, loader in self.test_loaders.items():
                l_acc, _ = Trainer.test(self.model, loader, device=self.device)
                pp(f'{phase}_acc:c:{l_acc}')
                if self.args.wandb and self.args.nth_repeat == 0:
                    wandb.log({f'acc/{phase}/sv_task': l_acc})
                self.results[phase][self.cur_epoch] = l_acc

    @staticmethod
    def test(model, loader, device='cpu'):
        label_correct = 0
        n_correct = 0
        total = len(loader.dataset)
        for _, (data, n, c_l) in enumerate(loader):
            data, n, c_l = data.to(device), n.to(device), c_l.to(device)
            n_logit, c_l_logit = model(data)
            _, c_l_pred = c_l_logit.max(dim=1)
            _, n_pred = n_logit.max(dim=1)
            label_correct += torch.sum(c_l_pred == c_l).item()
            n_correct += torch.sum(n_pred == n).item()
        return float(label_correct)/total, float(n_correct)/total


def iterate_args(args):
    args_list = list()
    for params in args.parameters:
        args.params = params
        args.usvt_weight = params[0]
        args.original_img_prob = params[1]

        s2ts = ms2st(args.domains, args.targets)
        for s2t in s2ts:
            args.source = s2t['s']
            args.target = s2t['t']
            for i in range(int(args.repeat_times)):
                args.nth_repeat = i
                args_list.append(copy.deepcopy(args))
    return args_list


def main():
    # This flag allows you to enable the inbuilt cudnn auto-tuner to
    # find the best algorithm to use for your hardware.
    torch.backends.cudnn.benchmark = True
    args = get_args()
    for args in iterate_args(args):
        output_dir = f'{args.output_dir}/{socket.gethostname()}/{args.experiment}/{args.network}/' + \
        '_'.join([str(_) for _ in args.params])+'/'
        writer = Writer(
            output_dir=output_dir,
            file=f'{args.source[0]}_{args.target}'
        )
        if args.redirect_to_file and args.redirect_to_file != 'null':
            print('redirect to ', output_dir+args.redirect_to_file)
            sys.stdout = open(output_dir+args.redirect_to_file, 'a')
        model = model_fns[args.network](
            num_usv_classes=args.num_usv_classes,
            num_classes=args.num_classes)
        if args.wandb and args.nth_repeat == 0:
            tags = [args.source[0] + '_' + args.target, "_".join([str(_) for _ in args.params])]
            wandb.init(project=f'{args.experiment}_{args.network}', tags=tags,
                       dir=dirname(__file__), config=args,
                       name=f'{"-".join([str(_) for _ in args.params])}-{args.source[0]}-{args.target}')

            wandb.watch(model, log='all')
        # data_loaders = get_DGR_data_loader(args.source, args.target, args.data_dir, args.val_size,
        #                                    args.original_img_prob, args.batch_size,
        #                                    args.max_num_s_img, args)
        data_loaders = get_DGIR_data_loader(args.source, args.target, args.data_dir, args.val_size,
                                           args.original_img_prob, args.batch_size,
                                           args.max_num_s_img, args)
        optimizer = get_optimizer(model, lr=args.learning_rate, train_all=args.train_all_param)
        scheduler = optim.lr_scheduler.StepLR(optimizer, int(args.epochs * .8))
        Trainer(args, model, data_loaders, optimizer, scheduler, writer)

        # torch.save(model.state_dict(), args.data_dir+'/cache/model.pkl')
        # wandb.save(args.data_dir+'/cache/model.pkl')
        if args.wandb and args.nth_repeat == 0:
            wandb.join()

if __name__ == "__main__":
    main()





