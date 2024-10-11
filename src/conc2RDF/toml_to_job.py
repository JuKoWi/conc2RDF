import tomllib

from .neural_network import NeuralNetwork
from .filereader import Directory, DataSetFromList
from .utils import merge_dictionaries

import torch.optim as optim
from torch import nn



class NNFromToml(NeuralNetwork):
    def __init__(self, num_outputs, my_toml):
        super().__init__(num_outputs=num_outputs)
        self.num_neurons = my_toml["neural_network"]["num_neurons"]
        self.optimizer = getattr(optim, my_toml["optimizer"]["type"])(self.parameters(), my_toml["optimizer"]["learning_rate"])
        self.criterion = getattr(nn, my_toml["neural_network"]["loss_function"])()


class ConfigLoader:
    def __init__(self, toml_filename):
        with open(toml_filename, "rb") as f:
            self.config = tomllib.load(f)



def do_the_job(toml_path=True):
    if toml_path is True:
        job_toml = {}
    else:
        job_toml = ConfigLoader(toml_path).config
    default_toml = ConfigLoader("/largedisk/julius_w/Development/conc2RDF/src/conc2RDF/defaulttoml.toml").config # TODO change path
    job_toml = merge_dictionaries(default_toml, job_toml)

    job_dir = Directory(job_toml["dataset"]["dirpath"])
    jobset = DataSetFromList(job_dir.get_relevant_files())
    train_conc = job_toml["learning"]["train_selection"]
    test_conc = job_toml["learning"]["test_selection"]
    train_data = jobset.get_subset_from_list(jobset.get_indices(train_conc))
    test_data = jobset.get_subset_from_list(jobset.get_indices(test_conc))
    num_runs = job_toml["learning"]["num_runs"]

    best_val_loss = float("inf")
    for run in range(num_runs):
        model = NNFromToml(train_data.get_output_size(), job_toml)
        model.train_network(train_data, test_data, job_toml["learning"]["epochs"])
        val_loss = model.val_losses[-1]
        print(f"Validation Loss for run {run+1}: {val_loss:.2e}")
        # Save the model if it has the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_model()
    print(f"Best validation loss: {best_val_loss:.2e}")

