import wandb
for x in range(10):
    wandb.init(project="runs-from-for-loop2", reinit=True)
    for y in range (100):
        wandb.log({"metric": x+y})
        wandb.run.save()
    wandb.join()