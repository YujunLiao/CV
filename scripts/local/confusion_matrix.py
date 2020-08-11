import os

import torch
from dgssr_trainer import model_fns, Trainer
from dl.data_loader.utils.get_p_l import get_p_l
from dl.dataset.ssr import SSRTest
from dl.dataset.rotation import RotTest
from dl.data_loader.utils.ds2dl import train_DL_fn, test_DL_fn
import numpy as np
# /home/giorgio/Files/pycharm_project/pytorch_interpreter/bin/python3.7 /home/giorgio/Files/pycharm_project/CV/scripts/local/confusion_matrix.py
from openpyxl import Workbook
import math
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set()
import matplotlib.pylab as plt

wb = Workbook()
# grab the active worksheet
ws = wb.active

class Container:
    def __int__(self):
        pass


#

def confusion_matrix(model, data_loader, n_classes, ust_classeds, device):
    label_correct = 0
    n_correct = 0
    total = len(data_loader.dataset)
    matrix = np.zeros((n_classes, n_classes))
    matrix_ust = np.zeros((ust_classeds, ust_classeds))
    for _, (data, n, c_l) in enumerate(data_loader):
        data, n, c_l = data.to(device), n.to(device), c_l.to(device)
        n_logit, c_l_logit = model(data)
        _, c_l_pred = c_l_logit.max(dim=1)
        _, n_pred = n_logit.max(dim=1)
        for (pred, real) in zip(c_l_pred, c_l):
            matrix[real][pred] += 1
        for (pred, real) in zip(n_pred, n):
            matrix_ust[real][pred] += 1

        label_correct += torch.sum(c_l_pred == c_l).item()
        n_correct += torch.sum(n_pred == n).item()
    print(float(label_correct) / total, float(n_correct) / total)

    print(matrix)
    print(matrix_ust)
    for i in range(matrix.shape[0]):
        matrix[i] = matrix[i]/sum(matrix[i])*100
    for i in range(matrix_ust.shape[0]):
        matrix_ust[i] = matrix_ust[i] / sum(matrix_ust[i]) * 100
    matrix = matrix.transpose()
    matrix_ust = matrix_ust.transpose()
    #print(matrix)
    #print(matrix_ust)
    return matrix, matrix_ust


def w(ws, i, j, value):
    ws[chr(ord('A') + j) + str(i + 1)] = value
def w_matrix(ws, begin_i, begin_j, matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            w(ws, i+begin_i, j+begin_j, matrix[i][j])

def heatmap2d(arr: np.ndarray):
    plt.imshow(arr, cmap='viridis')
    plt.colorbar()
    plt.show()



# 'photo', 'art_painting', 'cartoon'
project_path = '/home/giorgio/Files/pycharm_project/CV/'
model_path = 'output/era/spp2/caffenet/'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model_fns['caffenet'](
    num_usv_classes=4,
    num_classes=7)
model = model.to(device)
begin_i, begin_j = 0, 0
#for param in os.listdir(project_path+model_path):
for param in ['0.7_0.7_2']:
    for target in ['sketch']:
        for prob in [1, 0.25]:
            # torch.cuda.set_device(0)
            model.load_state_dict(torch.load(project_path + model_path+param+'/'+target))
            args = Container()
            args.image_size = 222
            test_paths, _, test_labels, _ = get_p_l(target, dir=project_path + 'data/test/')
            test_s_DS = RotTest(test_paths, test_labels, prob=prob, args=args)
            test_s_data_loader = test_DL_fn(test_s_DS, 128)

            # for i, (data, n, c_l) in enumerate(test_s_data_loader):
            #     data, n, c_l = data.to(device), n.to(device), c_l.to(device)
            model.eval()
            with torch.no_grad():
                # s_acc, us_acc = Trainer.test(model, test_s_data_loader, device=device)
                # print(j, target_i, s_acc, us_acc)
                matrix, matrix_ust = confusion_matrix(model, test_s_data_loader, 7, 4, device=device)
                #w(ws, begin_i, begin_j, param+target+str(prob))
                #begin_i += 1
                #w_matrix(ws, begin_i, begin_j, matrix)
                #w_matrix(ws, begin_i, begin_j+matrix.shape[0]+3, matrix_ust)
                #begin_i += matrix.shape[0]+3
                # for i in range(matrix.shape[0]):
                #     for j in range(matrix.shape[1]):
                #         w(ws, i, j, matrix[i][j])
                # for i in range(matrix_ust.shape[0]):
                #     for j in range(matrix_ust.shape[1]):
                #         w(ws, i, j+matrix.shape[0]+3, matrix[i][j])

                if prob == 1:
                    #plt.figure(figsize=(16, 16))
                    categories = ['dog', 'elep.', 'giraffe', 'guitar', 'horse', 'house', 'person']
                    chart = sns.heatmap(matrix, annot=True, fmt='.1f', cmap='Reds', cbar=True,)
                    chart.set_xticklabels(labels=categories)
                    chart.set_yticklabels(labels=categories)
                    plt.xlabel("Real")
                    plt.ylabel("Predict")
                    plt.title('Standard rotation')
                    plt.show()
                if prob == 0.25:
                    #plt.figure(figsize=(16, 16))
                    chart = sns.heatmap(matrix_ust, annot=True, fmt='.1f', cmap='Reds', cbar=True, )
                    plt.xlabel("Real")
                    plt.ylabel("Predict")
                    plt.title('Standard rotation')
                    plt.show()
                print(param)
                print(target)
                print(prob)
wb.save("./confusion_m.xlsx")

