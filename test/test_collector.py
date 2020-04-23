from dl.utils.collector import Recorder

def test_collector():
    col = Recorder()
    col.train.append({
        'acc_class': 0.1,
        'acc_r': 0.2,
        'loss_class': 3.5,
        'loss_r': 3.6,
        'epoch': 2,
        'mini_batch': 3})

    col.train.append({
        'acc_class': 0.1,
        'acc_r': 0.2,
        'loss_class': 3.5,
        'loss_r': 3.6,
        'epoch': 2,
        'mini_batch': 3})
    assert col.train.epoch == [2, 2]