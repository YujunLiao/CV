from random import sample
from os.path import dirname


def split(list_1, list_2, proportion=float(0)):
    """

    :param list_1: list of images paths
    :param list_2:  list of labels
    :param proportion: 0 < float < 1
    :return:
    """
    n = len(list_1)
    # indices is a list of index of (n * percent) samples from original data list.
    indices = sample(range(n), int(n * proportion))
    list_1_1 = [list_1[k] for k in indices]
    list_1_2 = [v for k, v in enumerate(list_1) if k not in indices]
    list_2_1 = [list_2[k] for k in indices]
    list_2_2 = [v for k, v in enumerate(list_2) if k not in indices]
    return [list_1_2, list_1_1, list_2_2, list_2_1]


def paths_and_labels_from(files):
    if isinstance(files, str):
        files = [files]
    paths = []
    labels = []
    for file in files:
        with open(file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.split(' ')
            paths.append(line[0])
            labels.append(int(line[1]))
    return [paths, labels]


def get_data(domains, dir=f'{dirname(__file__)}/../data/train/', val_size=float(0)):
    if isinstance(domains, str):
        domains = [domains]
    paths_1 = []
    paths_2 = []
    labels_1 = []
    labels_2 = []

    for domain in domains:
        p, l = paths_and_labels_from(dir+domain)
        # training_arguments.val_size refer to the percent of validation dataset, for example,
        # val_size=0.1 means 10% data is used for validation, 90% data is used for training.
        p_1, p_2, l_1, l_2 = split(p, l, val_size)
        paths_1 += p_1
        paths_2 += p_2
        labels_1 += l_1
        labels_2 += l_2

    return paths_1, paths_2, labels_1, labels_2