"""Create plots by loading the model.pth file for analysis.

Some plots need the path to the datasets as an additional argument.
"""

import matplotlib.pyplot as plt
import torch

from .neural_network import NeuralNetwork
from .rdf_dataset import RdfDataSet

"""TODO Find better solution to the following problem:
The Analyzer can not be operated with the model alone but only in combinaton with the dataset.
One needs to make sure that in the loops in show_predictions() and show_errors() the
prediction for the right concentration is plotted together with the data for the respective concentration
Otherwise the graphs could turn out wrong if a filename in the dataset is slightly changed.
"""


class Analyzer:
    """Create plots by loading the model.pth file for analysis.

    Some plots need the path to the datasets as an additional argument.
    """

    def __init__(self, model: NeuralNetwork):
        self.model: NeuralNetwork = model

    def get_dashboard(self):
        """plot training process information"""
        val_losses_np = [loss.cpu().numpy() for loss in self.model.val_losses]
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(self.model.train_losses, "o", ms=3, label="trainig")
        axs[1].plot(val_losses_np, "o", ms=3, label="testing")
        axs[0].set_ylabel("Training Loss")
        axs[1].set_ylabel("Validation Loss")
        axs[1].set_xlabel("$n_{\ Epoch}$")
        axs[0].semilogy()
        axs[1].semilogy()
        axs[0].legend()
        axs[1].legend()
        plt.savefig("training_plot.png")

    def show_errors(self, dataset: RdfDataSet):
        """Plot errors of the result for different concentrations."""
        self.inputs = dataset.inputs
        self.outputs = dataset.outputs
        self.model.eval()
        MSE = [None] * len(self.inputs)
        MAE = [None] * len(self.inputs)
        with torch.no_grad():
            for i in range(len(self.inputs)):
                X = self.inputs[i].to(self.model.device)
                pred = self.model(X)
                MSE[i] = (
                    torch.mean((pred - self.outputs[i].to(self.model.device)) ** 2)
                    .cpu()
                    .numpy()
                )
                MAE[i] = (
                    torch.mean(torch.abs(pred - self.outputs[i].to(self.model.device)))
                    .cpu()
                    .numpy()
                )
            inputs_np = [input.cpu().numpy() for input in self.inputs]
            plt.plot(inputs_np, MSE, "o", ms=3, label="mean square error")
            plt.plot(inputs_np, MAE, "o", ms=3, label="mean absolute error")
            plt.xlabel("c / 10 %")
            plt.ylabel("Error")
            plt.legend()
            plt.savefig("errorplot.png")

    def show_predictions(self, dataset: RdfDataSet):
        """Show the prediction rdf for different concentrations."""
        self.rvalues = dataset.rvalues
        self.inputs = dataset.inputs
        self.outputs = dataset.outputs
        self.model.eval()
        with torch.no_grad():
            for i in range(len(self.inputs)):
                X = self.inputs[i].to(self.model.device)
                pred = self.model(X)
                plt.plot(self.rvalues, pred.cpu(), "o", ms=3, label=f"{X.item()}")
                plt.plot(self.rvalues, self.outputs[i].to(self.model.device).cpu())
                plt.legend()
                plt.xlabel("r / $\\AA$")
                plt.ylabel("g(r)")
                plt.savefig(f"model_predictions_{X.item()}.png")
                plt.close()
