"""A program uses a vanilla neural network to map an RDF on a concentration."""

import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

# constants
DEBUG_MODE = False
DIRECTORY_PATH = "/largedisk/julius_w/Development/conc2RDF/training_data"


class DataDir:
    """Operations for whole directory."""

    def __init__(self, path: str) -> None:
        """Variables associated with the whole dataset/directory."""
        self.path = pathlib.Path(path)
        self.files = []
        self.allfiles = os.listdir(path)
        self.num_points = None
        self.r_values = None  # the r values for which  the datapoints are defined
        self.size = None
        self.data = None

    def get_relevant_files(self) -> None:
        """Only use data from rdf files."""
        for f in self.allfiles:
            newfile = File(f, self.path)
            if newfile.is_relevant():
                self.files.append(newfile)

    def extract_data(self) -> None:
        """Read all data from the files as well as concentrations from filename.

        Checks consistency of data in terms of number of bins.
        collects the size of the dataset
        all data (concentrations and rdf) as one sorted tensor
        the first element of each tensor line is concentation
        """
        for f in self.files:
            f.find_header()
            f.get_percentage()
            f.read_table()
        self.r_values = self.files[0].data[0]
        list_num_bins = [f.num_bins for f in self.files]
        assert all(
            element == list_num_bins[0] for element in list_num_bins
        ), "Not all training RDF have the same number of datapoints"
        self.num_points = list_num_bins[0]
        self.size = len(self.files)
        self.data = np.array([np.append(f.percentage, f.data[1]) for f in self.files])
        self.data = self.data[
            np.argsort(self.data[:, 0])
        ]  # sort the data concentrationwise
        self.data = torch.tensor(self.data, dtype=torch.float)


class File:
    """operations on single files."""

    def __init__(self, filename: str, directory: pathlib.Path) -> None:
        """Variables associated with a single file."""
        self.filename = filename
        self.percentage = None
        self.header = 0
        self.data = None
        self.path = directory / filename
        self.num_bins = None

    def get_percentage(self) -> None:
        """Read the butanol concentration from the filename."""
        if self.filename.startswith("rdf") and self.filename.endswith("bu.xvg"):
            self.percentage = float(self.filename[len("rdf") : -len("bu.xvg")])
        else:
            print("ERROR: Files do not match pattern")

    def find_header(self) -> None:
        """Check how many lines to skip when reading the data."""
        with open(self.path) as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith(("@", "#")):
                    self.header += 1
                else:
                    break

    def is_relevant(self) -> bool:
        """Check if the file contrains rdf-data."""
        return "rdf" in self.filename

    def read_table(self) -> None:
        """Read the rdf data for one file to np.array."""
        self.data = np.loadtxt(self.path, skiprows=self.header).T
        self.num_bins = np.shape(self.data)[1]


class NeuralNetwork(nn.Module):
    """Neural network that contains all the parameters."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.MSELoss()
    epochs = 2000

    def __init__(self, dataset: DataDir) -> None:
        """Create vanilla NN."""
        super().__init__()
        self.dataset = dataset
        self.losses = []
        self.validations_losses = []
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 64),
            # with 512 for all hidden layers loss and val_loss oscillate after some time
            nn.ReLU(),
            nn.Linear(64, 64),
            # 512 instead of 64, lower learning rate no oscillation but no better result
            nn.ReLU(),
            nn.Linear(64, dataset.num_points),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward propagation."""
        return self.linear_relu_stack(x)


    def train_step(self, train_set : list) -> float:
        """"Train the NN on a small sample of training data."""
        self.train()
        loss_value = 0
        for i in train_set:
            x_data = torch.tensor([self.dataset.data[i, 0]]).to(NeuralNetwork.device)
            y_data = self.dataset.data[i, 1:].to(NeuralNetwork.device)
            pred = self(x_data)
            loss = NeuralNetwork.loss_fn(pred, y_data)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            loss_value += loss.item()
        loss_value = loss_value / len(train_set)
        if DEBUG_MODE:
            print(f"loss: {loss_value:>7f}")
        return loss_value

    def test_step(self, test_set: list) -> float:
        """Compare prediction with real test-data."""
        self.eval()
        test_loss = 0
        with torch.no_grad():
            for i in test_set:
                x = torch.tensor([self.dataset.data[i, 0]]).to(NeuralNetwork.device)
                y = self.dataset.data[i, 1:].to(NeuralNetwork.device)
                pred = self(x)
                test_loss += NeuralNetwork.loss_fn(pred, y).item()
        test_loss /= len(test_set)
        return test_loss

    def train_test_loop(self, train_set: list, test_set: list):
        """perform several train and test steps and record the average loss"""
        for t in tqdm(range(NeuralNetwork.epochs)):
            if DEBUG_MODE:
                print(f"Epoch {t+1}\n -----------------------")
            avg_loss = self.train_step(train_set)
            self.losses.append(avg_loss)
            val_loss = self.test_step(test_set)
            self.validations_losses.append(val_loss)

    def get_dashboard(self):
        """plot """
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(self.losses, "o", ms=3, label="trainig")
        axs[1].plot(self.validations_losses, "o", ms=3, label="testing")
        axs[0].semilogy()
        axs[1].semilogy()
        axs[0].legend()
        axs[1].legend()
        plt.show()

    def save_model(self):
        torch.save(self, "model.pth")



def main() -> None:
    """Load the data, set up NN parameters, perform the training and save the NN."""
    new_dataset = DataDir(DIRECTORY_PATH)
    new_dataset.get_relevant_files()
    new_dataset.extract_data()

    #initialize NeuralNetwork
    my_network = NeuralNetwork(new_dataset)

    # set training choice
    choice = [0, 2, 4, 6, 9]  # indexes in the sorted concentration list
    for_test = [i for i in range(new_dataset.size) if i not in choice]
    print(choice)
    print(
        f"Training with the concentrations{[new_dataset.data[i, 0].item() for i in choice]}"
    )

    my_network.train_test_loop(choice, for_test)
    my_network.get_dashboard()
    my_network.save_model()


if __name__ == "__main__":
    main()
