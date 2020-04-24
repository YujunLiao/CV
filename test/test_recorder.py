from dl.utils.recorder import Recorder

def test_recorder():
    col = Recorder()
    col.train.append({
        'acc_class': 0.1,
        'acc_r': 0.2,
        'loss_class': 3.5,
        'loss_r': 3.6,
        'epoch': 2,
        'mini_batch': 3})

    Recorder.save(obj=col)

    col2 = Recorder.load()
    print(col2.train)
    # assert col.train == col2.train