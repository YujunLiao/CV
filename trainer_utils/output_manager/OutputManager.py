import numpy as np
import os

class OutputManager:
    def __init__(self, output_file_path, output_file_name):
        self.output_file_path = output_file_path
        self.output_file_name = output_file_name
        if not os.path.exists(output_file_path):
            os.makedirs(output_file_path)

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
