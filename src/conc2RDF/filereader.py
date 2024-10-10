import os
from pathlib import Path

import numpy as np
import torch

from .rdf_dataset import RdfDataSet


class DataSetFromList(RdfDataSet):
    """This subclass of RdfDataSet allows instances to be generated from a list if filepaths. It depends on the FromFile class"""

    def __init__(self, pathlist):
        self.inputs = None
        self.outputs = None
        self.get_from_pathlist(pathlist)

    def get_from_pathlist(self, pathlist):
        for path in pathlist:
            file = FileFactory.create_file_handler(path)
            file.get_percentage()
            file.read_table()
            self.add_item(file.input, file.output)


class FromFile:
    """Object that handles and contains information from one single file.

    Subclasses for other filetypes than xvg must implement
    get_percentage and read_table method to guarantee polymorphism
    """

    def __init__(self, path):
        self.path = Path(path)
        self.filename = Path(path).name
        self.input = None
        self.output = None
        self.num_bins = None
        self.rvalues = None

    def is_relevant(self) -> bool:
        """Check if the file contrains rdf-data."""
        return "rdf" in self.filename


class FileFactory:
    """Factory to create file handler objects based on the file type."""

    @staticmethod
    def create_file_handler(path: str) -> FromFile:
        pathpath = Path(path)
        if pathpath.suffix == ".xvg":
            return FromXVGFile(path)
        else:
            raise ValueError(f"ERROR: Invalid file format for file {path}")


class FromXVGFile(FromFile):
    """Subclass to read and contain information fom xvg file."""

    def __init__(self, path):
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
        """Check how many lines to skip when reading the data."""
        with open(self.path) as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith(("@", "#")):
                    self.header += 1
                else:
                    break

        """Read the rdf data for one file to np.array -> tourch tensor"""
        self.output = np.loadtxt(self.path, skiprows=self.header).T
        self.rvalues = self.output[0]
        self.num_bins = np.shape(self.output[1])
        self.output = np.expand_dims(self.output[1], axis=0)
        self.output = torch.tensor(self.output, dtype=torch.float)


class Directory:
    """A class that finds data containing files in a directory and
    returns the relevant files as a list of paths.
    """

    def __init__(self, path):
        self.pathpath = Path(path)
        self.path = path
        self.filepaths = []
        self.allfiles = os.listdir(path)

    def get_relevant_files(self):
        for f in self.allfiles:
            if f.endswith(".xvg"):
                newfile = FromXVGFile(self.pathpath / f)
                if newfile.is_relevant():
                    self.filepaths.append(self.path + "/" + f)
        return self.filepaths
