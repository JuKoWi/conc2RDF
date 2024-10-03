
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tqdm import tqdm

# constants
DEBUG_MODE = False

#first preprocess data

class Dataset_dir:
    def __init__(self, path):
        self.path = path
        self.files = []
        self.allfiles = os.listdir(path)
        self.num_points = None
        self.r_values = None # the r values for which  the datapoints are defined
        self.size = None
        self.data = None
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
        self.size = len(self.files)
        self.data = np.array([np.append(f.percentage, f.data[1]) for f in self.files])
        self.data = self.data[np.argsort(self.data[:,0])] #sort the data for the different concentrations concentrationwise
        self.data = torch.tensor(self.data, dtype = torch.float) 

        
class file:  
    def __init__(self, filename, directory):
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
print(new_dataset.data)

device = (
   "cuda"
   if torch.cuda.is_available()
   else "cpu"
)



class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(1, 64), #with 512 for all hidden laysers the loss and val_loss curve oscillate after some time
            nn.ReLU(),
            nn.Linear(64, 64), #with 512 instead of 64 and lover learning rate no oscillation but also no better results
            nn.ReLU(),
            nn.Linear(64, new_dataset.num_points)
        )

    def forward(self, x):
        return self.linear_relu_stack(x)

model = NeuralNetwork().to(device)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

def train(Dataset_dir, model, loss_fn, optimizer, show_each_batch=False):
    model.train()
    loss_value = 0
    for i in choice:    
        X = torch.tensor([Dataset_dir.data[i,0]]).to(device)
        y = Dataset_dir.data[i,1:].to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if show_each_batch:
            print(f"concentration {X.item()} has loss {loss.item()}")
        loss_value += loss.item()
    loss_value = loss_value / len(choice)
    if DEBUG_MODE: 
        print(f"loss: {loss_value:>7f}")
    return loss_value

def test(Dataset_dir, model, loss_fn):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for i in for_test:
            X = torch.tensor([Dataset_dir.data[i,0]]).to(device)
            y = Dataset_dir.data[i, 1:].to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
    test_loss /= len(for_test)
    return test_loss



#generate  random set for training
choice = [0, 3, 5, 7, 9] #indexes in the dorted concentration list 
for_test = [i for i in range(new_dataset.size) if i not in choice]
print(choice)
print([new_dataset.data[i,0] for i in choice])
print(for_test)

#perform training loop
print(device)
epochs = 3000 
losses = []
validations_losses = []
for t in tqdm(range(epochs)):
    if DEBUG_MODE: 
        print(f"Epoch {t+1}\n -----------------------")
    avg_loss = train(new_dataset, model, loss_fn, optimizer)
    losses.append(avg_loss)
    val_loss = test(new_dataset, model, loss_fn)
    validations_losses.append(val_loss)



fig, axs = plt.subplots(2,1)
axs[0].plot(losses, "o", ms =3, label = "trainig")
axs[1].plot(validations_losses, "o", ms = 3, label = "testing")
axs[0].semilogy()
axs[1].semilogy()
axs[0].legend()
axs[1].legend()
plt.show()

#print(losses[-1])
torch.save(model, "model.pth")
