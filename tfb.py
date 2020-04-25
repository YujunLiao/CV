import socket
from os.path import dirname

from tensorboardX import SummaryWriter
from dl.utils.recorder import Recorder


log_dir = f'{dirname(__file__)}/mod_saved/' \
          f'era/DG_rot/resnet18/0.4_0.8/'
rec_file = 'cartoon_art_painting.rec'
print(f'read{log_dir+rec_file}')
rec = Recorder.load(log_dir+rec_file)
print(rec.train.__dict__.keys())
# writer = SummaryWriter(log_dir=log_dir)
print()