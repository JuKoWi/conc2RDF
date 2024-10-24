"""Setup dataclass to contain configuration/input-parameters.

Parameters can come from a TOML file. The classes are nested and so are the TOML files.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Scheduler:
    """Configuration for the learning rate scheduler.

    Attributes:
        is_on (bool): Indicates if the scheduler is active.
        type (str): Type of the scheduler.
        mode (str): Mode of operation for the scheduler.
        factor (float): Factor by which the learning rate will be reduced.
        patience (int): Number of epochs with no improvement after which learning rate will be reduced.
        verbose (bool): If True, prints messages when the learning rate is reduced.
    """
    is_on: bool
    type: str
    mode: str
    factor: float
    patience: int
    verbose: bool


@dataclass
class Optimizer:
    """Configuration for the optimizer.

    Attributes:
        type (str): Type of optimizer to use.
        learning_rate (float): Learning rate for the optimizer. Default is 0.001.
    """
    type: str
    learning_rate: float = 0.001


@dataclass
class EarlyStopping:
    """Configuration for early stopping.

    Attributes:
        is_on (bool): Indicates if early stopping is enabled.
        patience (int): Number of epochs with no improvement after which training will stop.
        min_delta (float): Minimum change to consider as an improvement.
    """
    is_on: bool
    patience: int
    min_delta: float


@dataclass
class NetworkParms:
    """Network parameters for the neural network configuration.

    Attributes:
        num_neurons (List[int]): List containing the number of neurons in each layer.
        loss_function (str): Loss function to be used.
        optimizer (Optimizer): Optimizer configuration. Defaults to an instance of Optimizer.
    """
    num_neurons: List[int]
    loss_function: str
    optimizer: Optimizer = field(default_factory=Optimizer)


@dataclass
class Learning:
    """Learning configuration.

    Attributes:
        epochs (int): Number of epochs for training.
        print (bool): Indicates if training progress should be printed.
        train_selection (List[float]): Selection of training data.
        test_selection (List[float]): Selection of test data.
        num_runs (int): Number of runs for the training process.
        stopping (EarlyStopping): Early stopping configuration. Defaults to an instance of EarlyStopping.
        scheduler (Scheduler): Learning rate scheduler configuration. Defaults to an instance of Scheduler.
    """
    epochs: int
    print: bool
    train_selection: List[float]
    test_selection: List[float]
    num_runs: int
    stopping: EarlyStopping = field(default_factory=EarlyStopping)
    scheduler: Scheduler = field(default_factory=Scheduler)


@dataclass
class Dataset:
    """Dataset configuration.

    Attributes:
        files (bool): Indicates if files are used for the dataset.
        dirpath (str): Directory path where the dataset is located.
    """
    files: bool
    dirpath: str


@dataclass
class Config:
    """Main configuration class for the neural network.

    Attributes:
        nn (NetworkParms): Network parameters configuration.
        data (Dataset): Dataset configuration.
        learn (Learning): Learning configuration.
    """
    nn: NetworkParms
    data: Dataset
    learn: Learning
