import socket
from os.path import dirname

from tensorboardX import SummaryWriter
from dl.utils.recorder import Recorder


log_dir = f'{dirname(__file__)}/mod_saved/era/DG_rot/resnet18/0.4_0.8/'
rec_file = 'cartoon_art_painting.rec'
print(log_dir)
# writer = SummaryWriter(log_dir=log_dir)
rec = Recorder.load(log_dir+rec_file)
print(rec.train.__dict__.keys())
print()