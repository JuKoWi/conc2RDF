"""Define callbacks as subcalsses of Callbacks class to use in NeuralNetwork.train_network()."""

import torch.optim as optim


class Callbacks:
    """Template for Callback."""

    def __init__(self):
        self.stop_training = False

    def on_epoch_start(self):
        pass

    def on_epoch_end(self, current_loss: float):
        pass


class EarlyStoppingCallback(Callbacks):
    """Stop training run early by setting self.stop_training to true."""

    def __init__(self, patience=10, min_delta=0.001) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best_loss = float("inf")
        self.stop_training = False

    def on_epoch_end(self, current_loss: float) -> None:
        """Execute at end of epoch in NeuralNetwork.train_network()."""
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True


class LRScheduler(Callbacks):
    """Takes optim.lr_scheduler object as input."""

    def __init__(self, scheduler: optim.lr_scheduler):
        self.scheduler = scheduler
        self.stop_training = False

    def on_epoch_end(self, current_loss: float) -> None:
        """Execute .step() from troch.optim.lr_scheduler."""
        self.scheduler.step(current_loss)
