import tomllib

from conc2RDF import NeuralNetwork

import torch.optim as optim
from torch import nn



class NNFromToml(NeuralNetwork):
    def __init__(self, my_toml):
        super().__init__()
        self.optimizer = getattr(optim, my_toml.config["optimizer"]["type"])(my_toml["optimizer"]["learning_rate"])
        self.criterion = getattr(nn, my_toml.config["loss_function"])()







class ConfigLoader:
    def __init__(self, toml_filename):
        with open(toml_filename, "rb") as f:
            self.config = tomllib.load(f)
        
my_toml = ConfigLoader("defaulttoml.toml")
testnn = NNFromToml(10, my_toml)