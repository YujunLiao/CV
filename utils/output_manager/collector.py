import numpy as np
import os

import torch


class Collector:
    def __init__(self, output_dir='./output/', file='log'):
        self.output_dir = output_dir
        self.file = file
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except:
                print('create output file fail!!!!')

    def add(self, input_list):
        with open(self.output_dir + self.file, 'a') as f:
            for i in input_list:
                string = i
                if isinstance(i, list):
                    string = ""
                    for j in i:
                        string = string+j+' '
                f.write(string + '\n')

    def read(self):
        with open(self.output_dir + self.file, 'r') as f:
            return f.read()



