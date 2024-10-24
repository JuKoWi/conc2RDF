"""Define callbacks as subclasses of Callbacks class to use in NeuralNetwork.train_network()."""

import torch.optim as optim


class Callbacks:
    """Template for Callback.

    Attributes:
        stop_training (bool): Flag indicating whether to stop training.
    """

    def __init__(self):
        """Initialize the Callback class."""
        self.stop_training = False

    def on_epoch_start(self):
        """Execute at the start of each epoch."""
        pass

    def on_epoch_end(self, current_loss: float):
        """Execute at the end of each epoch.

        Args:
            current_loss (float): The current loss value at the end of the epoch.
        """
        pass


class EarlyStoppingCallback(Callbacks):
    """Stop training run early by setting self.stop_training to true.

    Attributes:
        patience (int): Number of epochs to wait before stopping.
        min_delta (float): Minimum change to consider as an improvement.
        wait (int): Counter for how many epochs without improvement.
        best_loss (float): Best loss value seen so far.
    """

    def __init__(self, patience=10, min_delta=0.001) -> None:
        """Initialize the EarlyStoppingCallback class.

        Args:
            patience (int, optional): Number of epochs to wait before stopping. Defaults to 10.
            min_delta (float, optional): Minimum change to consider as an improvement. Defaults to 0.001.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best_loss = float("inf")
        self.stop_training = False

    def on_epoch_end(self, current_loss: float) -> None:
        """Execute at the end of the epoch in NeuralNetwork.train_network().

        Args:
            current_loss (float): The current loss value at the end of the epoch.
        """
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stop_training = True


class LRScheduler(Callbacks):
    """Takes optim.lr_scheduler object as input.

    Attributes:
        scheduler (optim.lr_scheduler): The learning rate scheduler.
    """

    def __init__(self, scheduler: optim.lr_scheduler):
        """Initialize the LRScheduler class.

        Args:
            scheduler (optim.lr_scheduler): The learning rate scheduler to use.
        """
        self.scheduler = scheduler
        self.stop_training = False

    def on_epoch_end(self, current_loss: float) -> None:
        """Execute .step() from torch.optim.lr_scheduler.

        Args:
            current_loss (float): The current loss value at the end of the epoch.
        """
        self.scheduler.step(current_loss)
