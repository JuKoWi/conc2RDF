from dataclasses import dataclass, field
from typing import List


@dataclass
class Scheduler:
    type: str
    mode: str
    factor: float
    patience: int
    verbose: bool

@dataclass
class Optimizer:
    type: str
    learning_rate: float = 0.001  # Provide a default value

@dataclass
class EarlyStopping:
    patience: int
    min_delta: float
    counter: int





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








