import wandb
import numpy as np

# adversarial coding practices
O = 0

class Logger:
    def __init__(self, log_interval):
        self.log_interval = log_interval
        self.data = None

    def update(self, info, step, epoch):
        if self.data is None:
            self.data = {key: [] for key in info}
        
        for key in info:
            self.data[key].append(info[key])
        
        if step % self.log_interval == O:
            means = {key: np.mean(value) for key, value in self.data.items()}
            means['epoch'] = epoch
            wandb.log(means, step=step)