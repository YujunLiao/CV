import os
from time import time, strftime, localtime
import pickle

class Cache:
    def __init__(self, d=None):
        pass

    def append(self, d):
        for k in d.keys():
            if not hasattr(self, k):
                self.__dict__[k] = list()
            self.__dict__[k].append(d[k])

    def append_l(self, d):
        for k in d.keys():
            if not hasattr(self, k):
                self.__dict__[k] = list()
            self.__dict__[k] += d[k]


class Recorder():
    def __init__(self, data=None, output_dir='./output/', file='log.rec'):
        self.data = data if data else dict()
        assert isinstance(self.data, dict)
        self.train = Cache()
        self.val = Cache()
        self.test = Cache()
        self.model = Cache()

        self.output_dir = output_dir
        self.file = file
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def open(self):
        self.data['start_time'] = strftime("%Y-%m-%d %H:%M:%S", localtime())
        self.start_time = time()

    def close(self):
        assert self.start_time is not None
        self.data['duration'] = time() - self.start_time


    def save(self):
        with open(self.output_dir+self.file, 'wb') as f:
            pickle.dump(self, f)

    # @staticmethod
    # def save(obj, file='./output/log.rec'):
    #     with open(file, 'wb') as f:
    #         pickle.dump(obj, f)

    @staticmethod
    def load(file='./output/log.rec'):
        with open(file, 'rb') as f:
            return pickle.load(f)







# col = Recorder()
# col.train.append({
#     'acc_class': 0.1,
#     'acc_r': 0.2,
#     'loss_class': 3.5,
#     'loss_r': 3.6,
#     'epoch': 2,
#     'mini_batch': 3})
#
# col.save()
#
# col2 = Recorder.load()
# print()


        # def __init__(self, **kwargs):
        #     super().__init__()
        #     self.__dict__.update(kwargs)
        #
        # def __getitem__(self, index):
        #     b = Batch()
        #     for k in self.__dict__.keys():
        #         if self.__dict__[k] is not None:
        #             b.update(**{k: self.__dict__[k][index]})
        #     return b
        #
        # def update(self, **kwargs):
        #     self.__dict__.update(kwargs)
        #
        # def append(self, batch):
        #     assert isinstance(batch, Batch), 'Only append Batch is allowed!'
        #     for k in batch.__dict__.keys():
        #         if batch.__dict__[k] is None:
        #             continue
        #         if not hasattr(self, k) or self.__dict__[k] is None:
        #             self.__dict__[k] = batch.__dict__[k]
        #         elif isinstance(batch.__dict__[k], np.ndarray):
        #             self.__dict__[k] = np.concatenate([
        #                 self.__dict__[k], batch.__dict__[k]])
        #         elif isinstance(batch.__dict__[k], torch.Tensor):
        #             self.__dict__[k] = torch.cat([
        #                 self.__dict__[k], batch.__dict__[k]])
        #         elif isinstance(batch.__dict__[k], list):
        #             self.__dict__[k] += batch.__dict__[k]
        #         else:
        #             s = 'No support for append with type' \
        #                 + str(type(batch.__dict__[k])) \
        #                 + 'in class Batch.'
        #             raise TypeError(s)
        #
        #

