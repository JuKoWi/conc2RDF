import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from scipy.signal import savgol_filter

# Constants
DEBUG_MODE = False
DIRECTORY_PATH = "/largedisk/julius_w/Development/conc2RDF/training_data"

class DataDir:
    """Operations for whole directory."""

    def __init__(self, path: str) -> None:
            """Variables associated with the whole dataset/directory."""
            self.path = Path(path)
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
    """Operations on single files."""

    def __init__(self, filename : str, directory : Path) -> None:
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

# Custom EWMA Layer
class EWMALayer(nn.Module):
    """Exponentially Weighted Moving Average Layer."""

    def __init__(self, alpha=0.3):
        """Initialize the smoothing factor (alpha)."""
        super(EWMALayer, self).__init__()
        self.alpha = alpha

    def forward(self, x):
        """Apply EWMA to the input tensor."""
        result = torch.zeros_like(x)
        result[0] = x[0]  # First point remains the same
        for t in range(1, x.shape[0]):
            result[t] = self.alpha * x[t] + (1 - self.alpha) * result[t - 1]
        return result

class NeuralNetwork(nn.Module):
    """Neural network that contains all the parameters."""

    def __init__(self, dataset: DataDir) -> None:
        """Create vanilla NN and append EWMA as last layer."""
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, dataset.num_points),
        )
        self.ewma_layer = EWMALayer(alpha=0.3)  # Add EWMA layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward propagation and apply EWMA."""
        x = self.linear_relu_stack(x)
        x = self.ewma_layer(x)  # Apply EWMA smoothing to the output
        return x

# Backpropagation for training
def train(dataset: DataDir, model: NeuralNetwork, train_set: list):
    """Train the NN on a small sample of training data."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    model.train()
    loss_value = 0
    for i in train_set:
        x_data = torch.tensor([dataset.data[i, 0]]).to(device)
        y_data = dataset.data[i, 1:].to(device)
        pred = model(x_data)
        loss = loss_fn(pred, y_data)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss_value += loss.item()
    loss_value = loss_value / len(train_set)
    if DEBUG_MODE:
        print(f"loss: {loss_value:>7f}")
    return loss_value

# Compare prediction with testset
def test(dataset: DataDir, model: NeuralNetwork, test_set: list):
    """Compare prediction with real test-data."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = nn.MSELoss()
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i in test_set:
            x = torch.tensor([dataset.data[i, 0]]).to(device)
            y = dataset.data[i, 1:].to(device)
            pred = model(x)
            test_loss += loss_fn(pred, y).item()
    test_loss /= len(test_set)
    return test_loss

def main() -> None:
    """Load the data, set up NN parameters, perform the training, and save the NN."""
    new_dataset = DataDir(DIRECTORY_PATH)
    new_dataset.get_relevant_files()
    new_dataset.extract_data()

    # Set variables and parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NeuralNetwork(new_dataset).to(device)

    # Set training choice
    choice = [0, 2, 4, 6, 9]  # Indexes in the sorted concentration list
    for_test = [i for i in range(new_dataset.size) if i not in choice]
    print(choice)
    print(f"Training with the concentrations {[new_dataset.data[i, 0].item() for i in choice]}")

    # Perform training loop
    epochs = 1500
    losses = []
    validations_losses = []
    for t in tqdm(range(epochs)):
        if DEBUG_MODE:
            print(f"Epoch {t+1}\n -----------------------")
        avg_loss = train(new_dataset, model, choice)
        losses.append(avg_loss)
        val_loss = test(new_dataset, model, for_test)
        validations_losses.append(val_loss)

    # Plot optimization curve
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(losses, "o", ms=1, label="trainig")
    axs[1].plot(validations_losses, "o", ms=1, label="testing")
    axs[0].semilogy()
    axs[1].semilogy()
    axs[0].legend()
    axs[1].legend()
    plt.show()

    # Save the model
    torch.save(model, "model.pth")

if __name__ == "__main__":
    main()
