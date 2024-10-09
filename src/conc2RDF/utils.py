from pathlib import Path
import os
import numpy as np
import torch
import sys
import matplotlib.pyplot as plt

from .rdf_dataset import RdfDataSet
from .neural_network import NeuralNetwork


class DataSetFromList(RdfDataSet):
    """This subclass of RdfDataSet allows instances to be generated from a list if filepaths. It depends on the FromFile class"""

    def __init__(self, pathlist):
        self.inputs = None
        self.outputs = None
        self.get_from_pathlist(pathlist)

    def get_from_pathlist(self, pathlist):
        for path in pathlist:
            pathpath = Path(path)
            if pathpath.suffix == ".xvg":
                file = FromXVGFile(path)
            else:
                print("ERROR: Invalid file format")
            file.get_percentage()
            file.read_table()
            self.add_item(file.input, file.output)


class FromFile:
    """Object that handles and contains information from one single file.

    Subclasses for other filetypes than xvg must implement
    get_percentage and read_table method to guarantee polymorphism
    """

    def __init__(self, path):
        self.path = Path(path)
        self.filename = Path(path).name
        self.input = None
        self.output = None
        self.num_bins = None
        self.rvalues = None

    def is_relevant(self) -> bool:
        """Check if the file contrains rdf-data."""
        return "rdf" in self.filename


class FromXVGFile(FromFile):
    """Subclass to read and contain information fom xvg file."""

    def __init__(self, path):
        super().__init__(path)
        self.header = 0

    def get_percentage(self) -> None:
        """Read the butanol concentration from the filename."""
        if self.filename.startswith("rdf") and self.filename.endswith("bu.xvg"):
            percentage = float(self.filename[len("rdf") : -len("bu.xvg")])
            self.input = torch.tensor([[percentage]], dtype=torch.float)
        else:
            print("ERROR: Files do not match pattern")

    def read_table(self) -> None:
        """Check how many lines to skip when reading the data."""
        with open(self.path) as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith(("@", "#")):
                    self.header += 1
                else:
                    break

        """Read the rdf data for one file to np.array -> tourch tensor"""
        self.output = np.loadtxt(self.path, skiprows=self.header).T
        self.rvalues = self.output[0]
        self.num_bins = np.shape(self.output[1])
        self.output = np.expand_dims(self.output[1], axis=0)
        self.output = torch.tensor(self.output, dtype=torch.float)


class Directory:
    """A class that finds data containing files in a directory and
    returns the relevant files as a list of paths.
    """

    def __init__(self, path):
        self.pathpath = Path(path)
        self.path = path
        self.filepaths = []
        self.allfiles = os.listdir(path)

    def get_relevant_files(self):
        for f in self.allfiles:
            if f.endswith(".xvg"):
                newfile = FromXVGFile(self.pathpath / f)
                if newfile.is_relevant():
                    self.filepaths.append(self.path + "/" + f)
        return self.filepaths


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
