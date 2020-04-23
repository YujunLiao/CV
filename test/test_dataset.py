from os.path import dirname
from dl.dataset.base import BaseDataset
from dl.model.model import get_model

from dl.data_loader.DGR import get_DGR_data_loader


def test_data_loader():
    data_loader = get_DGR_data_loader('cartoon', 'photo', dirname(__file__)+'/../data/',
                                       0.1, 0.5, 4, 8)[0]



def test_model():
    model = get_model('resnet18',
                      jigsaw_classes=4,
                      classes=7)
    data_loader = get_DGR_data_loader('cartoon', 'photo', dirname(__file__) + '/../data/',
                                      0, 0.5, 6, 18)[0]
    for _, (data, n, label) in enumerate(data_loader):
        data, n, label = data.to('cpu'), n.to('cpu'), label.to('cpu')
        assert data.shape[0] == 6
        n_logit, l_logit = model(data)
        assert n_logit.shape[0] == 6
        _, n_pred = n_logit.max(dim=1)
        print(type(n_logit),type(n_pred), type(n))
        print(type(n[0].item()))
        print(n_pred==n)
        break







