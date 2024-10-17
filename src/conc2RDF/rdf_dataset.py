"""Clean class to contain dataset."""

import torch
from torch.utils.data import Dataset


class RdfDataSet(Dataset):
    """Store inputs (concentrations) and outputs (rdf values for bins).

    The rdf is only mapped to the respective concentration by the index in the inputs
    or outputs list.
    rvalue are the distance for which rdf values are provided.
    """

    def __init__(self, inputs, outputs) -> None:
        if len(inputs) != len(outputs):
            raise ValueError("Inputs and outputs must have the same length.")
        self.inputs = inputs
        self.outputs = outputs
        self.rvalues = None

    def get_indices(self, conc_list: list[int]) -> int:
        """Find the index of a specific concentration to access the data for this conc."""
        new_list = []
        for i in range(len(self.inputs)):
            if self.inputs[i] in conc_list:
                new_list.append(i)
        return new_list

    def get_subset_from_list(self, idx_list: list[int]):
        output_list = [self.outputs[i] for i in idx_list]
        input_list = [self.inputs[i] for i in idx_list]
        subset = RdfDataSet(input_list, output_list)
        subset.rvalues = self.rvalues
        return subset

    def get_output_size(self):
        """Get number of bins in rdf. Important for setup of NN."""
        return len(self.outputs[0])

    def __getitem__(self, index):
        """Find the specific """
        return self.inputs[index], self.outputs[index]

    def add_item(self, new_input, new_output):
        if self.inputs is None:
            # Initialize inputs and outputs with the shape of the first input/output
            self.inputs = new_input
            self.outputs = new_output
        if new_input not in self.inputs:
            self.inputs = torch.cat((self.inputs, new_input))
            self.outputs = torch.cat((self.outputs, new_output))
