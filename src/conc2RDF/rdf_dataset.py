"""Clean class to contain dataset."""

import torch
from torch.utils.data import Dataset


class RdfDataSet(Dataset):
    """Store inputs (concentrations) and outputs (rdf values for bins).

    The rdf is only mapped to the respective concentration by the index in the inputs
    or outputs list. rvalues are the distances for which rdf values are provided.

    Attributes:
        inputs (torch.Tensor): The input concentrations.
        outputs (torch.Tensor): The corresponding rdf values for the bins.
        rvalues (list[float] or None): The distances for which rdf values are provided.
    """

    def __init__(self, inputs: torch.Tensor, outputs: torch.Tensor) -> None:
        """Initialize the RdfDataSet with inputs and outputs.

        Args:
            inputs (torch.Tensor): The input concentrations.
            outputs (torch.Tensor): The corresponding rdf values for the bins.

        Raises:
            ValueError: If the lengths of inputs and outputs do not match.
        """
        if len(inputs) != len(outputs):
            raise ValueError("Inputs and outputs must have the same length.")
        self.inputs = inputs
        self.outputs = outputs
        self.rvalues = None

    def get_indices(self, conc_list: list[int]) -> list[int]:
        """Find the indices of specific concentrations to access the data.

        Args:
            conc_list (list[int]): List of concentrations to find indices for.

        Returns:
            list[int]: Indices of the inputs that match the specified concentrations.
        """
        new_list = []
        for i in range(len(self.inputs)):
            if self.inputs[i] in conc_list:
                new_list.append(i)
        return new_list

    def get_subset_from_list(self, idx_list: list[int]) -> 'RdfDataSet':
        """Create a new RdfDataSet instance for a list of concentrations.

        Args:
            idx_list (list[int]): List of indices to select from the dataset.

        Returns:
            RdfDataSet: A new RdfDataSet containing the selected inputs and outputs.
        """
        output_list = [self.outputs[i] for i in idx_list]
        input_list = [self.inputs[i] for i in idx_list]
        subset = RdfDataSet(input_list, output_list)
        subset.rvalues = self.rvalues
        return subset

    def get_output_size(self) -> int:
        """Get the number of bins in rdf.

        This is important for the setup of the neural network.

        Returns:
            int: The number of bins in the rdf outputs.
        """
        return len(self.outputs[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retrieve an item from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the input and the corresponding output.
        """
        return self.inputs[index], self.outputs[index]

    def add_item(self, new_input: torch.Tensor, new_output: torch.Tensor) -> None:
        """Add a new input-output pair to the dataset.

        There must be only one rdf for every concentration.

        Args:
            new_input (torch.Tensor): The new input concentration to add.
            new_output (torch.Tensor): The corresponding rdf value to add.
        """
        if self.inputs is None:
            # Initialize inputs and outputs with the shape of the first input/output
            self.inputs = new_input
            self.outputs = new_output
        if new_input not in self.inputs:
            self.inputs = torch.cat((self.inputs, new_input))
            self.outputs = torch.cat((self.outputs, new_output))
