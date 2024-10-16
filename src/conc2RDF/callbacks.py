import torch.optim as optim


class Callbacks:
    def __init__(self):
        self.stop_training = False

    def on_epoch_start(self):
        pass

    def on_epoch_end(self, current_loss: float):
        pass


class EarlyStoppingCallback(Callbacks):
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best_loss = float("inf")
        self.stop_training = False
        

    def on_epoch_end(self, current_loss: float):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
            return False
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True

class LRScheduler(Callbacks):
    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.stop_training = False
    
    def on_epoch_end(self, current_loss: float):
        self.scheduler.step(current_loss)
        