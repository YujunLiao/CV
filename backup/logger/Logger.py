from time import time

from os.path import join, dirname

from .tf_logger import TFLogger

_log_path = join(dirname(__file__), '../logs')


# high level wrapper for tf_logger.TFLogger
class Logger():
    def __init__(self, args, update_frequency=10):
        self.current_epoch = 0
        self.epochs_number = args.epochs
        self.current_iter = 0

        self.last_update_time = time()
        self.start_time = time()
        self._clean_epoch_stats()
        self.update_f = update_frequency

        folder_name, log_name = self.get_log_relative_path_from_args(args)
        log_path = join(_log_path, folder_name, log_name)
        if args.tf_logger:
            self.tf_logger = TFLogger(log_path)
            # print("Saving to %s" % log_path)
            # print('--------------------------------------------------------')
        else:
            self.tf_logger = None

    def new_epoch(self, learning_rates):
        self.current_epoch += 1
        self.last_update_time = time()
        # TODO(lyj):
        self.learning_rates = learning_rates
        # print("New epoch - lr: %s" % ", ".join([str(lr) for lr in self.learning_rates]))
        self._clean_epoch_stats()
        if self.tf_logger:
            for n, v in enumerate(self.learning_rates):
                self.tf_logger.scalar_summary("aux/lr%d" % n, v, self.current_iter)

    def log(self, it, iters, losses, samples_right, total_samples):
        self.current_iter += 1
        loss_string = ", ".join(["%s : %.3f" % (k, v) for k, v in losses.items()])
        for k, v in samples_right.items():
            past = self.epoch_stats.get(k, 0.0)
            self.epoch_stats[k] = past + v
        self.total += total_samples
        acc_string = ", ".join(["%s : %.2f" % (k, 100 * (v / total_samples)) for k, v in samples_right.items()])
        if it % self.update_f == 0:
            print("%d/%d of epoch %d/%d %s - acc %s [bs:%d]" % (it, iters, self.current_epoch, self.epochs_number, loss_string, acc_string, total_samples))
            # update tf log
            if self.tf_logger:
                for k, v in losses.items(): self.tf_logger.scalar_summary("trainer/loss_%s" % k, v, self.current_iter)

    def _clean_epoch_stats(self):
        self.epoch_stats = {}
        self.total = 0

    def log_test(self, phase, accuracies):
        print("Accuracies on %s: " % phase + ", ".join(["%s : %.2f" % (k, v * 100) for k, v in accuracies.items()]))
        print('--------------------------------------------------------')
        if self.tf_logger:
            for k, v in accuracies.items(): self.tf_logger.scalar_summary("%s/acc_%s" % (phase, k), v, self.current_iter)

    def save_best(self, val_test, best_test):

        print("It took %g" % (time() - self.start_time))
        print('--------------------------------------------------------')
        if self.tf_logger:
            for x in range(10):
                self.tf_logger.scalar_summary("best/from_val_test", val_test, x)
                self.tf_logger.scalar_summary("best/max_test", best_test, x)

    @staticmethod
    def get_log_relative_path_from_args(args):
        folder_name = "%s_to_%s" % ("-".join(sorted(args.source)), args.target)
        if args.folder_name:
            folder_name = join(args.folder_name, folder_name)
        log_name = "eps%d_bs%d_lr%g_class%d_jigClass%d_jigWeight%g" % (args.epochs, args.batch_size, args.learning_rate, args.n_classes,
                                                                   4, args.unsupervised_task_weight)
        # if args.ooo_weight > 0:
        #     name += "_oooW%g" % args.ooo_weight
        if args.train_all:
            log_name += "_TAll"
        if args.bias_whole_image:
            log_name += "_bias%g" % args.bias_whole_image
        if args.classify_only_original_img:
            log_name += "_classifyOnlySane"
        if args.TTA:
            log_name += "_TTA"
        try:
            log_name += "_entropy%g_jig_tW%g" % (args.entropy_weight, args.target_weight)
        except AttributeError:
            pass
        if args.suffix:
            log_name += "_%s" % args.suffix
        log_name += "_%d" % int(time() % 1000)
        return folder_name, log_name
