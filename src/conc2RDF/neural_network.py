"""Sets up NeuralNetwork with network specific parameters/architecture.

The architecture can be given as a list of hidden layer neuron numbers.
Training method that saves train and validation losses in lists.
The train and test datasets are given as separate arguments and for callbacks a separate
class is used.
"""

import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm

from .callbacks import *
from .rdf_dataset import RdfDataSet


class NeuralNetwork(nn.Module):
    """Class that represents the Neural Network.

    Attributes:
        num_neurons (list[int]): List that describes the architecture of the network.
        lr (float): Initial learning rate.
        criterion (nn.Module): Loss function used for training.
        train_losses (list[float]): Loss in training set after each epoch.
        val_losses (list[float]): Loss in validation set after each epoch.
    """

    def __init__(
        self, num_outputs: int = 190, lr: float = 0.001, num_neurons: list[int] = [50]
    ) -> None:
        """Initialize the NeuralNetwork with given properties.

        Args:
            num_outputs (int): The number of output neurons.
            lr (float): The initial learning rate.
            num_neurons (list[int]): List of hidden layer neuron counts.
        """
        super().__init__()
        self.num_neurons = num_neurons
        input_size = 1
        layers = []
        for i in self.num_neurons:
            layers.append(nn.Linear(input_size, i))
            layers.append(nn.ReLU())
            input_size = i

        layers.append(nn.Linear(input_size, num_outputs))
        self.network = nn.Sequential(*layers)

        self.lr = lr
        self.criterion = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_losses = []
        self.val_losses = []
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward propagation through the network.

        Args:
            x (torch.Tensor): Input tensor for the forward pass.

        Returns:
            torch.Tensor: Output tensor from the network.
        """
        return self.network(x)

    def train_network(
        self,
        train_data: RdfDataSet,
        test_data: RdfDataSet,
        optimizer: optim,
        epochs: int = 1000,
        print_progress: bool = False,
        callbacks: list[Callbacks] = None,
    ):
        """Train the neural network.

        For the given number of epochs, backpropagation is performed with the train_data.
        Validation losses are recorded for test_data. Both kinds of losses are averaged
        over the samples from train_data and test_data. Callbacks from the list are 
        evaluated at the end of each epoch.

        Args:
            train_data (RdfDataSet): Dataset for training.
            test_data (RdfDataSet): Dataset for testing/validation.
            optimizer (optim.Optimizer): Optimizer to be used for training.
            epochs (int): Number of training epochs.
            print_progress (bool): Whether to print progress during training.
            callbacks (list[Callbacks], optional): List of callback objects to be evaluated 
                at the end of each epoch.
        """
        self.rvalues = train_data.rvalues  # for later use in analyzer
        self.callbacks = callbacks or []
        self.optimizer = optimizer

        progress_bar = tqdm(range(epochs), leave=True)
        for epoch in progress_bar:
            avg_loss = 0.0
            avg_val_loss = 0.0

            # Train part
            for x, y_ref in train_data:
                self.train()
                x = x.to(self.device)
                y_ref = y_ref.to(self.device)
                y_pred = self(x)
                loss = self.criterion(y_pred, y_ref)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss_value = loss.item()
                avg_loss += loss_value

            # Test part
            self.eval()
            with torch.no_grad():
                for x, y_ref in test_data:
                    x = x.to(self.device)
                    y_ref = y_ref.to(self.device)
                    y_pred = self(x)
                    avg_val_loss += self.criterion(y_pred, y_ref)

            avg_loss /= len(train_data[0])
            avg_val_loss /= len(test_data[0])
            self.train_losses.append(avg_loss)
            self.val_losses.append(avg_val_loss)

            if print_progress:
                if (epoch + 1) % 10 == 0:
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    progress_bar.set_description(
                        f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}, lr: {current_lr:.6e}"
                    )
            for callback in self.callbacks:
                callback.on_epoch_end(self.val_losses[-1])
            if any([callback.stop_training for callback in self.callbacks]):
                break

    def save_model(self):
        """Save the current state of the model.

        This method is used in the main.py file to save the model's state.
        """
        torch.save(self, "model.pth")
