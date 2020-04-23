from dl.dataset.base import BaseDataset

def test_dataset():
    _dataset = BaseDataset([1, 2, 3], [1, 2, 3])
    assert _dataset.argv == ([1, 2, 3], [1, 2, 3])
    assert len(_dataset) == 3
    assert _dataset[0] == (1, 1)
