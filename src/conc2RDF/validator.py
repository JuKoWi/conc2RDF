import logging
import os

from .config import Config


def validate_input(config: Config):
    try:
        if not os.path.exists(config.data.dirpath):
            raise FileNotFoundError(
                "The relative path to the directory for dataset files does not exist"
            )
    except FileNotFoundError as e:
        print(f"\033[91mERROR: {e}\033[0m")
        exit()

    try:
        for i in config.learn.test_selection + config.learn.train_selection:
            valid_conc = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
            if i not in valid_conc:
                raise ValueError(
                    "Concentration selection has to be in range 1 - 100 % with steps of 10"
                )
    except ValueError as e:
        print(f"\033[91mERROR: {e}\033[0m")
        exit()

    """The following block does not throw an error if empty list is provided. This simply means no hidden layer"""
    try:
        if not (
            isinstance(config.nn.num_neurons, list)
            and all(isinstance(i, int) for i in config.nn.num_neurons)
        ):
            raise ValueError(
                "Number of each layer of neurons has to be provided as list of integers"
            )
    except ValueError as e:
        print(f"\033[91mERROR: {e}\033[0m")
        exit()

    """Keep the following list updated according to the pytorch documentation"""
    optim_list = [
        "Adadelta",
        "Adafactor",
        "Addagrad",
        "Adam",
        "AdamW",
        "SparseAdam",
        "Adamax",
        "ASGD",
        "LBFGS",
        "NAdam",
        "RAdam",
        "RMSprop",
        "Rprop",
        "SGD",
    ]
    try:
        if not config.nn.optimizer.type in optim_list:
            logging.error("Adam error")
            raise ValueError(
                f"Choose an optimizer according to the pytorch documentation: {optim_list}"
            )

    except ValueError as e:
        print(f"\033[91mERROR: {e}\033[0m")
        exit()

    scheduler_list = [
        "CosineAnnealingWarmRestarts",
        "OneCycleLR",
        "CyclicLR",
        "ReduceLROnPlateau",
        "SequentialLR",
        "ChainedScheduler",
        "CosineAnnealingLR",
        "PolynomialLR",
        "ExponentialLR",
        "LinearLR",
        "ConstantLR",
        "MultiStepLR",
        "StepLR",
        "MultiplicativeLR",
        "LambdaLR",
        "LRScheduler",
    ]
    try:
        if not config.learn.scheduler.type in scheduler_list:
            raise ValueError(
                f"Choose a scheduler from torch.optim.lr_scheduler {scheduler_list}"
            )
    except ValueError as e:
        print(f"\033[91mERROR: {e}\033[0m")
        exit()
