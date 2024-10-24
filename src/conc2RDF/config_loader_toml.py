"""Create object of Config class from toml."""

import tomllib

from .config import (
    Config,
    Dataset,
    EarlyStopping,
    Learning,
    NetworkParms,
    Optimizer,
    Scheduler,
)
from .utils import merge_dictionaries


def load_toml(toml_filename):
    """Translation between the toml file and the Config class must be hardcoded."""
    config_dict = {}
    if toml_filename != "default":
        with open(toml_filename, "rb") as f:
            config_dict = tomllib.load(f)
    with open("./conc2RDF/default_config.toml", "rb") as f:
        default_dict = tomllib.load(f)
    config_dict = merge_dictionaries(default_dict, config_dict)

    scheduler = Scheduler(
        config_dict["learning"]["scheduler"]["is_on"],
        config_dict["learning"]["scheduler"]["type"],
        config_dict["learning"]["scheduler"]["mode"],
        config_dict["learning"]["scheduler"]["factor"],
        config_dict["learning"]["scheduler"]["patience"],
        config_dict["learning"]["scheduler"]["verbose"],
    )
    optimizer = Optimizer(
        config_dict["neural_network"]["optimizer"]["type"],
        config_dict["neural_network"]["optimizer"]["learning_rate"],
    )
    stop = EarlyStopping(
        config_dict["learning"]["early_stopping"]["is_on"],
        config_dict["learning"]["early_stopping"]["patience"],
        config_dict["learning"]["early_stopping"]["min_delta"],
    )

    nn = NetworkParms(
        config_dict["neural_network"]["num_neurons"],
        config_dict["neural_network"]["loss_function"],
        optimizer,
    )
    learn = Learning(
        config_dict["learning"]["epochs"],
        config_dict["learning"]["print_progress"],
        config_dict["learning"]["train_selection"],
        config_dict["learning"]["test_selection"],
        config_dict["learning"]["num_runs"],
        stop,
        scheduler,
    )
    data = Dataset(
        config_dict["dataset"]["filelist"],
        config_dict["dataset"]["dirpath"],
    )

    return Config(
        nn,
        data,
        learn,
    )
