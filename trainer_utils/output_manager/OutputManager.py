import numpy as np
import os

import torch


class OutputManager:
    def __init__(self, output_file_path, output_file_name):
        self.output_file_path = output_file_path
        self.output_file_name = output_file_name
        if not os.path.exists(output_file_path):
            try:
                os.makedirs(output_file_path)
            except:
                print('create file fail!!!!')


    def write_to_output_file(self, input_list):
        with open(self.output_file_path+self.output_file_name, 'a') as f:
            for i in input_list:
                string = i
                if isinstance(i, list):
                    string = ""
                    for j in i:
                        string = string+j+' '

                f.write(string + '\n')




    def read_from_output_file(self):
        with open(self.output_file_path+self.output_file_name, 'r') as f:
            return f.read()

    @staticmethod
    def print(data):
        print('----------------------------------------------')
        if isinstance(data, str):
            print(data)
        if isinstance(data, list):
            for _ in data:
                print(_)
        if isinstance(data, dict):
            for _ in data.items():
                if isinstance(_[1], torch.utils.data.Dataset):
                    print(_[0], len(_[1]))
                else:
                    print(_[0], _[1])











# output_manager = OutputManager(
#     '/home/giorgio/Files/pycharm_project/DG_rotation/trainer_utils/output_manager/output_file/'+\
#     '12/1243/')
#
# output_manager.write_to_output_file([
#     'hello1\n',
#     'hello2\n',
#     'hello3\n',
# ])
#
# print(output_manager.read_from_output_file())
