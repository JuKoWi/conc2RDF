import matplotlib.pyplot as plt
import torch

from .neural_network import NeuralNetwork
from .rdf_dataset import RdfDataSet

"""TODO Find better solution to the following problem:
The Analyzer can not be operated with the model alone but only in combinaton with the dataset.
One needs to make sure that in the loops in show_predictions() and show_errors() the
prediction for the right concentration is plotted together with the data for the respective concentration
Otherwise the graphs could turn out wrong if a filename in the dataset is slightly changed."""


class Analyzer:
    def __init__(self, model: NeuralNetwork):
        self.model: NeuralNetwork = model
        self.inputs = None
        self.ouputs = None
        self.rvalues = model.rvalues

    # TODO make dashboard class for plot of losses and RDF plots
    def get_dashboard(self):
        """plot training process information"""
        val_losses_np = [loss.cpu().numpy() for loss in self.model.val_losses]
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(self.model.train_losses, "o", ms=3, label="trainig")
        axs[1].plot(val_losses_np, "o", ms=3, label="testing")
        axs[0].semilogy()
        axs[1].semilogy()
        axs[0].legend()
        axs[1].legend()
        plt.savefig("training_plot.png")

    def show_errors(self, dataset: RdfDataSet):
        # TODO: mean like in paper
        """Plot errors of the resultfor different concentrations."""
        self.inputs = dataset.inputs
        self.outputs = dataset.outputs
        self.model.eval()
        MSE = [None] * len(self.inputs)
        MAE = [None] * len(self.inputs)
        with torch.no_grad():
            for i in range(len(self.inputs)):
                X = self.inputs[i].to(self.model.device)
                pred = self.model(X)
                MSE[i] = torch.mean((pred - self.outputs[i].to(self.model.device)) ** 2).cpu().numpy()
                MAE[i] = torch.mean(
                    torch.abs(pred - self.outputs[i].to(self.model.device))
                ).cpu().numpy()
            inputs_np = [input.cpu().numpy() for input in self.inputs]
            plt.plot(inputs_np, MSE, "o", ms=3, label="mean square error")
            plt.plot(inputs_np, MAE, "o", ms=3, label="mean absolute error")
            plt.legend()
            plt.savefig("errorplot.png")
            plt.close()

    def show_predictions(self, dataset: RdfDataSet):
        """Show the prediction rdf for different concentrations."""
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
                plt.savefig(f"model_predictions_{X.item()}.png")
                plt.close()
