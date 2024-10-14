import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm

from .rdf_dataset import RdfDataSet


class EarlyStoppingCallback:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best_loss = float("inf")

    def is_early_stop(self, current_loss: float, epoch: int):
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
            return False
        else:
            self.wait += 1
            if self.wait >= self.patience:
                stop_training = True
                print(f"Early stopping triggered at epoch {epoch}")
                return True


class NeuralNetwork(nn.Module):
    def __init__(self, num_outputs=190, lr=0.001, num_neurons=[50]):
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

        """For callbacks:
        TODO: Should I rather inlcude the possibility for callbacks in the NeuralNetwork 
        parent class or in the NNfromToml subclass? By defining the train_network() method 
        differently for NNFromToml I could use polymorphism
        """
        self.scheduler = None
        self.early_stopping = False
        self.early_stopping_patience = None
        self.early_stopping_min_delta = None
        self.early_stopping_counter = 0
        self.best_val_loss = float("inf")

        self.lr = lr
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_losses = []
        self.val_losses = []
        self.rvalues = None
        """TODO find better solution for the following problem:
        although not really a part of the NN, the rvalues have to be stored in the NN,
        to make sure, that by just loading the NN into the Ananlyzer class the result-RDF can be plotted.
        This impairs the single responsibility principle"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def train_network(
        self,
        train_data: RdfDataSet,
        test_data: RdfDataSet,
        epochs=1000,
        print_progress=False,
        #config: GodObjectWithEverything
    ):
        self.rvalues = train_data.rvalues
        progress_bar = tqdm(range(epochs), leave=True)

        #early_stopping_callback = EarlyStoppingCallback(patience=10, mi)
        for epoch in progress_bar:
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
                    progress_bar.set_description(
                        f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, Val Loss: {avg_val_loss:.6f}"
                    )

            # if not self.scheduler == None:
            #     self.scheduler.step(avg_val_loss)
            # if self.early_stopping:
            #     if self._check_early_stopping(avg_val_loss):
            #         print(f"Early stopping at epoch {epoch + 1} due to no improvement.")
            #         break

            #if early_stopping_callback.is_early_stop():
            #    break

    def save_model(self):
        torch.save(self, "model.pth")

    def _check_early_stopping(self, current_val_loss):
        """Check if early stopping should be triggered."""
        if current_val_loss < self.best_val_loss - self.early_stopping_min_delta:
            self.best_val_loss = current_val_loss
            self.early_stopping_counter = 0
        else:
            self.early_stopping_counter += 1

        if self.early_stopping_counter >= self.early_stopping_patience:
            return True
        return False
