### outline: a NN is created as a class. One attribute fo the class is the folder with the training data.
### planned methods: calculationg the loss function, optimizing the loss function, saving the parameters, creating an RDF from the input concentration

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# constants
DEBUG_MODE = True

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
        self.x_data = torch.tensor([[f.percentage] for f in self.files]) #it seems important that every element is a list on its own, otherwise dimensionality error
        self.y_data = np.array([f.data[1] for f in self.files])
        self.y_data = torch.tensor(self.y_data, dtype=torch.float)
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


device = (
   "cuda"
   if torch.cuda.is_available()
   else "cpu"
)

print(f"Using {device} device")


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, new_dataset.num_points) 

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


model = NeuralNetwork().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())


# some random data as well as some debugging
trainlist = torch.tensor([[10.0], [20.0] , [40.0], [60.0], [90.0]])#why does it only work if I take the list like this
trainlisty = torch.rand(100,5)
print(trainlisty[:,1])
print(new_dataset.x_data[1])
print(new_dataset.y_data[0])



#generate  random set for training
choice = np.random.choice(range(new_dataset.size), 6, replace = False)
for_test = [i for i in range(new_dataset.size) if i not in choice]
print(choice)
print(for_test)

def train(Dataset_dir, model, loss_fn, optimizer):
    model.train()
    loss_value = 0
    for i in choice:    
        X = Dataset_dir.x_data[i].to(device)
        y = Dataset_dir.y_data[i].to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if DEBUG_MODE:
            print(loss.item())
        loss_value += loss.item()
    loss_value = loss_value / len(choice) 
    print(f"loss: {loss_value:>7f}")

epochs = 3 
print(device)
for t in range(epochs): 
    print(f"Epoch {t+1}\n -----------------------")
    train(new_dataset, model, loss_fn, optimizer)

print("Done!")