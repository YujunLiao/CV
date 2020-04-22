from random import sample
from os.path import dirname


def split(list_1, list_2, proportion=None):
    """

    :param list_1: list of images paths
    :param list_2:  list of labels
    :param proportion: 0 < float < 1
    :return:
    """
    if proportion == None:
        return list_1, list_2
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


def get_data(source_domains, target_domain,
             dir=f'{dirname(__file__)}/data/', val_size = 0.1):
    assert isinstance(source_domains, list)
    assert isinstance(target_domain, str)

    train_paths = []
    val_paths = []
    train_labels = []
    val_labels = []

    for domain in source_domains:
        paths, labels = paths_and_labels_from(f'{dir}/train/{domain}')
        # training_arguments.val_size refer to the percent of validation dataset, for example,
        # val_size=0.1 means 10% data is used for validation, 90% data is used for training.
        paths_1, paths_2, labels_1, labels_2 = split(
            paths,
            labels,
            val_size)
        train_paths += paths_1
        val_paths += paths_2
        train_labels += labels_1
        val_labels += labels_2

    test_paths, test_labels = paths_and_labels_from(f'{dir}/train/{target_domain}')

    return train_paths, train_labels, val_paths, val_labels, test_paths, test_labels