
from .neural_network import NeuralNetwork
from .file_reader import Directory, DataSetFromList

import torch.optim as optim
from torch import nn


class NNFromToml(NeuralNetwork):
    def __init__(self, num_outputs, my_toml):
        super().__init__(num_outputs=num_outputs)
        self.num_neurons = my_toml["neural_network"]["num_neurons"]
        self.optimizer = getattr(optim, my_toml["neural_network"]["optimizer"]["type"])(
            self.parameters(), my_toml["neural_network"]["optimizer"]["learning_rate"],
        )
        self.criterion = getattr(nn, my_toml["neural_network"]["loss_function"])()
        self.scheduler = getattr(
            optim.lr_scheduler, my_toml["learning"]["scheduler"]["type"],
        )(
            self.optimizer,
            mode=my_toml["learning"]["scheduler"]["parameters"]["mode"],
            factor=my_toml["learning"]["scheduler"]["parameters"]["factor"],
            patience=my_toml["learning"]["scheduler"]["parameters"]["patience"],
            verbose=my_toml["learning"]["scheduler"]["parameters"]["verbose"],
        )
        self.early_stopping = "early_stopping" in my_toml["learning"]
        self.early_stopping_patience = my_toml["learning"]["early_stopping"]["patience"]
        self.early_stopping_min_delta = my_toml["learning"]["early_stopping"]["min_delta"]
        self.early_stopping_counter = my_toml["learning"]["early_stopping"]["counter"]
        print(self.early_stopping_patience)






