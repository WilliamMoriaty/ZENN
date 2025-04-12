#!/usr/bin/env python
# coding: utf-8

# In[147]:


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


# In[148]:


# 网络结构
import torch
import torch.nn as nn
import numpy as np

class Two_NN(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.width = width
        self.fc1 = nn.Linear(1,width)
        #self.fc2 = nn.Linear(width,width)
        #self.fc3 = nn.Linear(width,width)
        self.fc2 = nn.Linear(width,1,bias=False)
        self.act1 = nn.ReLU()
        self.act2 = nn.Tanh()
    
    def forward(self,x):
        y = self.fc1(x)
        y = self.act2(y)
        y = self.fc2(y)
        #y = self.act2(y)
        #y = self.fc3(y)
        # y = self.act2(y)
        # y = self.fc4(y)
        return y


# In[149]:


# Define CZ model
import torch.nn.init as init
class CoupledModel(nn.Module):
    def __init__(self, width, num_networks=6, kb=1.0):
        super(CoupledModel, self).__init__()
        self.num_networks = num_networks
        self.sub_networks = nn.ModuleList([Two_NN(width) for _ in range(num_networks)])  # 6 个 Two_NN 子网络
        self.kb = kb  # Boltzmann 常数
    def forward(self, x):
        # 计算每对网络的未归一化输出
        T = x[:, -1].unsqueeze(1)  # Shape: [batch_size, 1]
        logits = []
        values = []
        sub_network_outputs = []  # Store all sub-network outputs
        for i in range(self.num_networks):
            net_out = self.sub_networks[i](T)  
            sub_network_outputs.append(net_out)  # Store individual outputs

        # Convert list of tensors to a single tensor [batch_size, num_networks]
        sub_network_outputs = torch.cat(sub_network_outputs, dim=1)

        for i in range(self.num_networks // 2):
            net1_out = self.sub_networks[2 * i](T)  # net(2i)
            net2_out = self.sub_networks[2 * i + 1](T)  # net(2i+1)^2
            net2_out = net2_out**2  # net(2i+1)**2 ensure S_k positive
            logit = torch.exp(-(net1_out - T * net2_out)/(self.kb*T)-(net2_out/(5.0*self.kb))**2)  # exp(-(net1 - T * net2)/(kb*T)- (net2/sqrt(r)*kb)**2)
            
            logit = torch.exp(-(net1_out - T * net2_out)/(self.kb*T))  # exp(-(net1 - T * net2)/(kb*T))
            value = net1_out - T * net2_out  # net1 - T * net2
            logits.append(logit)
            values.append(value)

        logits = torch.cat(logits, dim=1)  # Shape [batch_size, 3]
        values = torch.cat(values, dim=1)  # Shape [batch_size, 3]

        # Softmax-like normalization
        softmax_weights = logits / torch.sum(logits, dim=1, keepdim=True)

        return softmax_weights, sub_network_outputs  # Return both final output and sub-network outputs
    


# In[150]:


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


# In[151]:


# load data

import pandas as pd


# Load CSV file as a Pandas DataFrame
df = pd.read_csv("Multiclass_data.csv",header = None)

# Convert DataFrame to PyTorch tensor (float type)
X_T_train = torch.tensor(df.values, dtype=torch.float64)

print("X_T_train shape:", X_T_train.shape)
print(X_T_train)


print(X_T_train[:, -1].unsqueeze(1))


# In[ ]:


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


#Define temperature range
Tmin = 1
Tmax = 8.0
Nf = 100
T = torch.linspace(Tmin, Tmax, Nf).unsqueeze(1).to(device)  # Ensure tensor is on the correct device

# Compute softmax probabilities for all T values in a single pass
softmax_weights, sub_outputs = model(T)  # Pass entire batch

# Convert to NumPy array
prob_array = softmax_weights.detach().cpu().numpy()  # Shape: [Nf, num_classes]

# Save to file
np.savetxt("CZ_multiclass_prob.txt", prob_array, fmt="%.6f")

print("Probabilities saved successfully! Shape:", prob_array.shape)

