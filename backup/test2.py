import wandb
for x in range(3):
    wandb.init(project="runs-from-for-loop2", name='t2', reinit=True)
    for y in range (100):
        wandb.log({"tr/metric": x+y})
        wandb.log({"test/ss": x + y})
    wandb.join()