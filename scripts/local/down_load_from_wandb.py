import wandb
from os.path import dirname, abspath
api = wandb.Api()
run = api.run("yujun-liao/CV-scripts_local/32vh4rub")
for file in run.files():
    if 'images' in file.name:
        print(file.name)
        dir = dirname(__file__)+'/../../output/'
        print(abspath(dir))
        file.download(root=dir, replace=True)