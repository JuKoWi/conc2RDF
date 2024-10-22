"""Setup dataclass to contain configuration/input-parameters.

Paramters can come from toml file. The classes are nested and so are the toml files.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Scheduler:
    is_on: bool
    type: str
    mode: str
    factor: float
    patience: int
    verbose: bool


@dataclass
class Optimizer:
    type: str
    learning_rate: float = 0.001


@dataclass
class EarlyStopping:
    is_on: bool
    patience: int
    min_delta: float


@dataclass
class NeuralNetwork:
    num_neurons: List[int]
    loss_function: str
    optimizer: Optimizer = field(default_factory=Optimizer)


@dataclass
class Learning:
    epochs: int
    print: bool
    train_selection: List[float]
    test_selection: List[float]
    num_runs: int
    stopping: EarlyStopping = field(default_factory=EarlyStopping)
    scheduler: Scheduler = field(default_factory=Scheduler)


@dataclass
class Dataset:
    files: bool
    dirpath: str


@dataclass
class Config:
    nn: NeuralNetwork
    data: Dataset
    learn: Learning
