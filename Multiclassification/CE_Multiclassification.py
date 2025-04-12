#!/usr/bin/env python
# coding: utf-8

# In[180]:


import torch
import torch.autograd as autograd         # computation graph
from torch import Tensor                  # tensor node in the computation graph
import torch.nn as nn                      # neural networks
import torch.optim as optim               # optimizers e.g. gradient descent, ADAM, etc.

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker
from sklearn.model_selection import train_test_split

import numpy as np
import time
import scipy.io

#Set default dtype to double
torch.set_default_dtype(torch.double)

#PyTorch random number generator
torch.manual_seed(1234)      

# Random number generators in other libraries
np.random.seed(1234)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

if device == 'cuda': 
    print(torch.cuda.get_device_name()) 


# In[181]:


# 网络结构
import torch
import torch.nn as nn
import numpy as np

class Two_NN(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.width = width
        self.fc1 = nn.Linear(1,width)
        self.fc2 = nn.Linear(width,width)
        self.fc3 = nn.Linear(width,width)
        self.fc4 = nn.Linear(width,width)
        self.fc5 = nn.Linear(width,width)
        self.fc6 = nn.Linear(width,width)
        self.fc7 = nn.Linear(width,3,bias=False)
        self.act1 = nn.ReLU()
        self.act2 = nn.Tanh()
    
    def forward(self,x):
        y = self.fc1(x)
        y = self.act2(y)
        y = self.fc2(y)
        y = self.act2(y)
        y = self.fc3(y)
        y = self.act2(y)
        y = self.fc4(y)
        y = self.act2(y)
        y = self.fc5(y)
        y = self.act2(y)
        y = self.fc6(y)
        y = self.act2(y)
        y = self.fc7(y)
        
        return y


# In[182]:


import torch.nn.functional as F

class CoupledModel(nn.Module):
    def __init__(self, width, num_networks=1, kb=1.0):
        super(CoupledModel, self).__init__()
        self.num_networks = num_networks
        self.sub_networks = nn.ModuleList([Two_NN(width) for _ in range(num_networks)])  # 
        self.kb = kb  # Boltzmann constant

    def forward(self, x):
        T = x[:, -1].unsqueeze(1)  # [batch_size, 1]

        # Output of the sub-network: shape [batch_size, 3]
        sub_outputs = self.sub_networks[0](T)/(self.kb*T)  # Assume one subnetwork only

        # Softmax along last dimension (3 logits per sample)
        softmax_weights = F.softmax(sub_outputs, dim=1)  # Shape: [batch_size, 3]

        return softmax_weights, sub_outputs


# In[183]:


import torch
import torch.nn as nn

class CrossLoss(nn.Module):
    def __init__(self):
        super(CrossLoss, self).__init__()

    def forward(self, softmax_weights, target):
        """
        Compute the custom loss using inner product.

        Parameters:
        - softmax_weights: [batch_size, num_classes] - Probability distribution output.
        - target: [batch_size, num_classes] - True probability distribution.

        Returns:
        - total_loss: Custom loss value.
        """
        #  Ensure target and predictions have the same shape
        assert softmax_weights.shape == target.shape, \
            f"Shape mismatch: softmax_weights {softmax_weights.shape}, target {target.shape}"

        #  Ensure `target` is float type
        target = target.to(dtype=softmax_weights.dtype)

        #  Compute inner product row-wise (avoid log(0))
        softmax_weights = torch.clamp(softmax_weights, min=1e-9)  # Prevent log(0)
        inner_product = -torch.sum(torch.log(softmax_weights) * target, dim=1)  # Shape: [batch_size]

        #  Compute mean loss
        total_loss = torch.mean(inner_product)  # Shape: scalar

        return total_loss


# In[184]:


# load data

import pandas as pd


# Load CSV file as a Pandas DataFrame
df = pd.read_csv("Multiclass_data.csv",header = None)

# Convert DataFrame to PyTorch tensor (float type)
X_T_train = torch.tensor(df.values, dtype=torch.float64)

print("X_T_train shape:", X_T_train.shape)
print(X_T_train)
target = X_T_train [:, :-1] 
target = target.long()
print("target shape:", target)



# In[185]:


import torch
import torch.optim as optim
import torch.nn as nn

# Initialize the model
width = 8  # Two_NN hidden layer width
model = CoupledModel(width).to(device)

# Create optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Define loss function
criterion = CrossLoss()  # Ensure using CrossLoss

# Training loop without mini-batches
num_epochs = 20001

X_T_train = X_T_train.to(device)  # Move entire dataset to device
target = X_T_train[:, :-1]  # Convert to integer class indices

for epoch in range(num_epochs):
    optimizer.zero_grad()  # Clear previous gradients

    # Forward pass
    softmax_weights, sub_outputs = model(X_T_train)

    # Compute loss
    loss = criterion(softmax_weights, target)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{num_epochs}] - Loss: {loss.item():.10f}")


# In[ ]:


# Define temperature range
Tmin = 1.0
Tmax = 8.0
Nf = 100
T = torch.linspace(Tmin, Tmax, Nf).unsqueeze(1).to(device)  # Ensure tensor is on the correct device

# Compute softmax probabilities for all T values in a single pass
softmax_weights, sub_outputs = model(T)  # Pass entire batch

# Convert to NumPy array
prob_array = softmax_weights.detach().cpu().numpy()  # Shape: [Nf, num_classes]

# Save to file
np.savetxt("CE_multiclass_prob.txt", prob_array, fmt="%.6f")

print("Probabilities saved successfully! Shape:", prob_array.shape)

