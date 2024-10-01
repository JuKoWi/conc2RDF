### outline: a NN is created as a class. One attribute fo the class is the folder with the training data.
### planned methods: calculationg the loss function, optimizing the loss function, saving the parameters, creating an RDF from the input concentration

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np



#first preprocess data

class Dataset_dir:
    def __init__(self, path):
        self.path = path
        self.files = []
        self.allfiles = os.listdir(path)
        self.alldata = {}
        self.num_points = None
        self.x_data = None
        self.y_data = None
        self.r_values = None # the r values for which  the datapoints are defined
        self.size = None
    def get_relevant_files(self):
        for f in self.allfiles:
            newfile = file(f, self.path)
            if newfile.is_relevant():
                self.files.append(newfile)    
    def extract_data(self):
        for f in self.files:
            f.find_header()
            f.get_percentage()
            f.read_table()
        self.r_values = self.files[0].data[0]
        list_num_bins = [f.num_bins for f in self.files]
        assert all(element == list_num_bins[0] for element in list_num_bins), "Not all training RDF have the same number of datapoints"
        self.num_points = list_num_bins[0]
        self.x_data = torch.tensor([f.percentage for f in self.files])
        self.y_data = np.array([f.data[1] for f in self.files])
        self.y_data = torch.tensor(self.y_data)
        self.size = len(self.files)
        

        
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

    
new_dataset = Dataset_dir("/largedisk/julius_w/Development/conc2RDF/training_data")
new_dataset.get_relevant_files()
new_dataset.extract_data()
print(new_dataset.x_data.size())
print(new_dataset.y_data.size())


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32,new_dataset.num_points)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
model = NeuralNetwork().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

def train(Dataset_dir, model, loss_fn, optimizer):
    model.train()
    for i in range(Dataset_dir.size):
        print(Dataset_dir.x_data[0])
        X, y = Dataset_dir.x_data[i].to(device), Dataset_dir.y_data[i].to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loss, current = loss.item(), (i + 1) * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n -----------------------")
    train(new_dataset, model, loss_fn, optimizer)
print("Done!")