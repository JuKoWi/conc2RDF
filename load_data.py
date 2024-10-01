import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import main

class file:  
    def __init__(self, filename, directory):
        self.directory = directory
        self.filename = filename
        self.percentage = None 
        self.header = 0
        self.data = None
        self.path = os.path.join(directory, filename)
        self.num_bins = None

    def get_percentage(self):
        if self.filename.startswith("rdf") and self.filename.endswith("bu.xvg"):
            self.percentage = float(self.filename[len("rdf"):-len("bu.xvg")])
        else:
            print("ERROR: Files do not match pattern") 

    def find_header(self):
        with open(self.path, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.startswith("@") or line.startswith("#"):
                    self.header += 1
                else:
                    break

    def is_relevant(self):
        if "rdf" in self.filename:
            return True
    
    def read_table(self):
        self.data = np.loadtxt(self.path, skiprows = self.header).T
        self.num_bins = np.shape(self.data)[1]


class rdf_dataset(Dataset):
    def __init__(self, dir, transform = None):
        self.dir = dir
        self.transform = transform
        arrays = []
        for f in os.listdir(dir):
            newfile = file(f, self.dir)
            if newfile.is_relevant():
                newfile.find_header()
                newfile.get_percentage()
                newfile.read_table()
                subarray = np.append([newfile.percentage], newfile.data[1])
                arrays.append(subarray)
        self.arrays = np.vstack(arrays)         

    def __len__(self):
        return len(self.arrays)

    def __getitem__(self, rdfx):
        sample = {"conc" : self.arrays[rdfx,0], "rdf" : self.arrays[rdfx,1:]}
        return sample

        
    





test = rdf_dataset("/largedisk/julius_w/Development/conc2RDF/training_data")

