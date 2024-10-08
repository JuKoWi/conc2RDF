"""Where is the best place to ensure that the tensors in RdfDataSet have dtype=torch.float
TODO : Where should I keep the information of the r-values to which the rdf-output values are assigned? Would make sense to me to keep it in the RdfDataSet class
Do I have to add a .to(device) somewhere to actually use the gpu?

flags:
    -p path to data containing directory
"""

from pathlib import Path
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset


class RdfDataSet(Dataset):
    def __init__(self, inputs, outputs):
        if len(inputs) != len(outputs):
            raise ValueError("Inputs and outputs must have the same length.")
        self.inputs = inputs
        self.outputs = outputs

    def get_indices(self, conc_list):
        new_list = []
        for i in range(len(self.inputs)):
            if self.inputs[i] in conc_list:
                new_list.append(i)
        return new_list

    def get_subset_from_list(self, idx_list):
        output_list = [self.outputs[i] for i in idx_list]
        input_list = [self.inputs[i] for i in idx_list]
        return RdfDataSet(input_list, output_list)

    def get_output_size(self):
        return len(self.outputs[0])

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]

    def add_item(self, new_input, new_output):
        if self.inputs is None:
            # Initialize inputs and outputs with the shape of the first input/output
            self.inputs = new_input
            self.outputs = new_output
        if new_input not in self.inputs:
            self.inputs = torch.cat((self.inputs, new_input))
            self.outputs = torch.cat((self.outputs, new_output))


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
        self.output = torch.tensor([self.output[1]], dtype=torch.float)


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


class NeuralNetwork(nn.Module):
    def __init__(self, num_outputs: int, lr=0.001):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_outputs),
        )
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_losses = []
        self.val_losses = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def train_network(self, train_data: RdfDataSet, test_data: RdfDataSet, epochs=1000, print_progress=False):
        #TODO insert tqdm bar
        for epoch in range(epochs):
            avg_loss = 0.0
            avg_val_loss = 0.0

            """train part"""
            for x, y_ref in train_data:
                self.train()
                y_pred = self(x)
                loss = self.criterion(y_pred, y_ref)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss_value = loss.item()
                avg_loss += loss_value

            """test_part"""
            self.eval()
            with torch.no_grad():
                for x, y_ref in test_data:
                    y_pred = self(x)
                    avg_val_loss += self.criterion(y_pred, y_ref)

            avg_loss /= len(train_data[0])
            avg_val_loss /= len(test_data[0])
            self.train_losses.append(avg_loss)
            self.val_losses.append(avg_val_loss)

            if print_progress:
                if (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, Validation Loss: {avg_val_loss:.6f}"
                    )

    def save_model(self):
        torch.save(self, "model.pth")



"""some miscellaneous functions"""

def is_intersection_train_test(train_dataset, test_dataset):
    for cons1 in train_dataset.inputs:
        for cons2 in test_dataset.inputs:
            if cons1 == cons2:
                return True
    return False

def get_input_args():
    arg_dict = {}
    for i,arg in enumerate(sys.argv):
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


def main():
    arg_dict = get_input_args()
    newdir = Directory(arg_dict["dir_path"])
    newset = DataSetFromList(newdir.get_relevant_files())
    print(newset.inputs)
    train_conc = [10.0, 30.0, 50.0, 70.0, 90.0]
    test_conc = [20.0, 40.0, 60.0, 80.0, 100.0]
    train_data = newset.get_subset_from_list(newset.get_indices(train_conc))
    test_data = newset.get_subset_from_list(newset.get_indices(test_conc))
    model = NeuralNetwork(train_data.get_output_size(), lr=0.001)
    print(model.device)
    model.train_network(train_data, test_data, 2000)
    model.save_model()
    get_dashboard(model)



if __name__ == "__main__":
    main()
