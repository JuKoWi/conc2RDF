"""Read Data from files to create RdfDataSet instance.

FileData to contain information from a single file. Subclasses for different file formats
with a factory object.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

from .rdf_dataset import RdfDataSet


class FileData(ABC):
    """Object that handles and contains information from one single file.

    Subclasses for file types other than xvg must implement
    get_percentage and read_table methods to guarantee polymorphism.

    Attributes:
        path (Path): The path to the file.
        filename (str): The name of the file.
        input (torch.Tensor): The input concentration read from the file.
        output (torch.Tensor): The rdf values read from the file.
        num_bins (int or None): The number of bins/datapoints for the rdf.
        rvalues (list[float] or None): The distances for which rdf values are provided.
    """

    def __init__(self, path: str) -> None:
        """Characterize the file by its path, name, and input/output data.

        Args:
            path (str): The path to the file being read.
        """
        self.path = Path(path)
        self.filename = self.path.name
        self.input = None
        self.output = None
        self.num_bins = None
        self.rvalues = None

    def is_relevant(self) -> bool:
        """Check if the file contains rdf data.

        Returns:
            bool: True if the file is relevant (contains rdf data), False otherwise.
        """
        return "rdf" in self.filename

    @abstractmethod
    def get_percentage(self):
        """Get the percentage/input for the file."""
        pass

    @abstractmethod
    def read_table(self):
        """Get the rdf/output for the file."""
        pass


class XVGFile(FileData):
    """Subclass to read and contain information from an xvg file."""

    def __init__(self, path: str) -> None:
        """Initialize the XVGFile object.

        Args:
            path (str): The path to the xvg file.
        """
        super().__init__(path)
        self.header = 0

    def get_percentage(self) -> None:
        """Read the butanol concentration from the filename."""
        if self.filename.startswith("rdf") and self.filename.endswith("bu.xvg"):
            percentage = float(self.filename[len("rdf") : -len("bu.xvg")])
            self.input = torch.tensor([[percentage]], dtype=torch.float)
        else:
            print("ERROR: Files do not match pattern")

    def read_table(self) -> None:
        """Read the rdf data from the file and convert it to a torch tensor."""
        self.get_header()
        self.output = np.loadtxt(self.path, skiprows=self.header).T
        self.rvalues = self.output[0]
        self.num_bins = np.shape(self.output[1])
        self.output = np.expand_dims(self.output[1], axis=0)
        self.output = torch.tensor(self.output, dtype=torch.float)

    def get_header(self) -> None:
        """Determine how many lines to skip when reading the data."""
        with open(self.path) as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith(("@", "#")):
                    self.header += 1
                else:
                    break


class FileFactory:
    """Factory to create file handler objects based on the file type."""

    @staticmethod
    def create_file_handler(path: str) -> FileData:
        """Create a file handler based on the file's extension.

        Args:
            path (str): The path to the file.

        Returns:
            FileData: An instance of a file handler (e.g., XVGFile) for the given file.

        Raises:
            ValueError: If the file format is invalid.
        """
        pathpath = Path(path)
        if pathpath.suffix == ".xvg":
            return XVGFile(path)
        else:
            raise ValueError(f"ERROR: Invalid file format for file {path}")


class DataSetFromList(RdfDataSet):
    """Subclass of RdfDataSet allowing instances to be generated from a list of file paths."""

    def __init__(self, pathlist: list[str]) -> None:
        """Initialize the dataset from a list of file paths.

        Args:
            pathlist (list[str]): List of file paths to read from.
        """
        self.inputs = None
        self.outputs = None
        self.get_from_pathlist(pathlist)

    def get_from_pathlist(self, pathlist: list[str]) -> None:
        """Populate the dataset from the provided list of file paths.

        Args:
            pathlist (list[str]): List of file paths to read from.
        """
        for path in pathlist:
            file = FileFactory.create_file_handler(path)
            file.get_percentage()
            file.read_table()
            self.add_item(file.input, file.output)
            self.rvalues = file.rvalues
            """TODO: The line above is a quick and dirty solution.
            The rvalues have to be stored in the RdfDataSet class, but the current
            implementation has two downsides: 
            On one hand, it does not check if the rvalues for all samples of the dataset are consistent. 
            On the other hand, to add the feature that datasets that store the rvalue not in every rdf file 
            but in a separate file can be used, the code has to be rewritten (not good extendability).
            """


class Directory:
    """A class that finds data-containing files in a directory.

    It returns the relevant files as a list of paths.
    """

    def __init__(self, path: str) -> None:
        """Initialize the Directory object.

        Args:
            path (str): The path to the directory to search for files.
        """
        self.pathpath = Path(path)
        self.path = path
        self.filepaths = []
        self.allfiles = os.listdir(path)

    def get_relevant_files(self) -> list[str]:
        """Get the relevant files from the directory.

        This method should also work if there are non-data files in the directory.

        Returns:
            list[str]: A list of paths to the relevant files.
        """
        for f in self.allfiles:
            if f.endswith(".xvg"):
                newfile = XVGFile(self.pathpath / f)
                if newfile.is_relevant():
                    self.filepaths.append(self.path + "/" + f)
        return self.filepaths
