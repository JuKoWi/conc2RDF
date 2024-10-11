
import torch
import torch.optim as optim
from torch import nn
from tqdm import tqdm

from .rdf_dataset import RdfDataSet


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


        self.lr = lr
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_losses = []
        self.val_losses = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def train_network(
        self,
        train_data: RdfDataSet,
        test_data: RdfDataSet,
        epochs=1000,
        print_progress=False,
    ):
        for epoch in tqdm(range(epochs)):
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
                        f"\r Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}, Validation Loss: {avg_val_loss:.6f}",
                        end=""
                    )
    def save_model(self):
        torch.save(self, "model.pth")
