
import sys

import matplotlib.pyplot as plt

from .neural_network import NeuralNetwork

"""some miscellaneous functions"""


def is_intersection_train_test(train_dataset, test_dataset):
    for cons1 in train_dataset.inputs:
        for cons2 in test_dataset.inputs:
            if cons1 == cons2:
                return True
    return False


def get_input_args():
    arg_dict = {}
    for i, arg in enumerate(sys.argv):
        if arg.startswith("-p"):
            arg_dict["dir_path"] = sys.argv[i + 1]
    return arg_dict


# TODO make dashboard class for plot of losses and RDF plots
def get_dashboard(network: NeuralNetwork):
    """plot"""
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(network.train_losses, "o", ms=3, label="trainig")
    axs[1].plot(network.val_losses, "o", ms=3, label="testing")
    axs[0].semilogy()
    axs[1].semilogy()
    axs[0].legend()
    axs[1].legend()
    plt.show()
