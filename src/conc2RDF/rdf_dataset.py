"""TODO write ravlues to RdfDataSet class"""

from torch.utils.data import Dataset
import torch


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
