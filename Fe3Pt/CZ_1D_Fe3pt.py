#!/usr/bin/env python
# coding: utf-8

# In[10]:


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


# In[11]:


import torch
import numpy as np
import matplotlib.pyplot as plt



def boltzmann_probability(V,x_vals,T, kb=0.1):
    P = torch.exp(-V / (kb*T))  # Compute unnormalized probability
    P /= torch.trapz(P.squeeze(), x_vals.squeeze())  # Normalize
    return P





# In[ ]:


# 网络结构
import torch
import torch.nn as nn
import numpy as np

class Two_NN(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.width = width
        self.fc1 = nn.Linear(2,width)
        self.fc2 = nn.Linear(width,width)
        # self.fc3 = nn.Linear(width,width)
        self.fc3 = nn.Linear(width,1,bias=False)
        self.act1 = nn.ReLU()
        self.act2 = nn.Tanh()
    
    def forward(self,x):
        y = self.fc1(x)
        y = self.act2(y)
        y = self.fc2(y)
        y = self.act2(y)
        y = self.fc3(y)
        # y = self.act2(y)
        # y = self.fc4(y)
        return y


# In[ ]:


# Define CZ model
import torch.nn.init as init
class CoupledModel(nn.Module):
    def __init__(self, width, num_networks=24, kb=0.1):
        super(CoupledModel, self).__init__()
        self.num_networks = num_networks
        self.sub_networks = nn.ModuleList([Two_NN(width) for _ in range(num_networks)])  # 6 个 Two_NN 子网络
        self.kb = kb  # Boltzmann 常数
    def forward(self, x):
        # 计算每对网络的未归一化输出
        T = x[:, 1].unsqueeze(1)  # Shape: [batch_size, 1]
        logits = []
        values = []
        sub_network_outputs = []  # Store all sub-network outputs
        for i in range(self.num_networks):
            net_out = self.sub_networks[i](x)  
            sub_network_outputs.append(net_out)  # Store individual outputs

        # Convert list of tensors to a single tensor [batch_size, num_networks]
        sub_network_outputs = torch.cat(sub_network_outputs, dim=1)

        for i in range(self.num_networks // 2):
            net1_out = self.sub_networks[2 * i](x)  # net(2i)
            net2_out = (self.sub_networks[2 * i + 1](x))  # net(2i+1)^2
            net2_out = net2_out**2  # net(2i+1)**2 ensure S_k positive
            logit = torch.exp(-(net1_out - T * net2_out)/(self.kb*T)-(net2_out/(5.0*self.kb))**2)  # exp(-(net1 - T * net2)/(kb*T)- (net2/sqrt(r)*kb)**2)
            value = net1_out - T * net2_out  # net1 - T * net2
            logits.append(logit)
            values.append(value)

        logits = torch.cat(logits, dim=1)  # Shape [batch_size, 3]
        values = torch.cat(values, dim=1)  # Shape [batch_size, 3]

        # Softmax-like normalization
        softmax_weights = logits / torch.sum(logits, dim=1, keepdim=True)

        # Compute the first term: weighted sum of (net1 - T * net2)
        weighted_values = torch.sum(softmax_weights * values, dim=1, keepdim=True)

        # Compute the second term: weighted sum of log probabilities
        log_probs = torch.log(softmax_weights + 1e-9)  # Avoid log(0)
        weighted_log_probs = torch.sum(softmax_weights * log_probs, dim=1, keepdim=True)

        # Final output
        output = weighted_values + self.kb * T * weighted_log_probs
        #output = (1/self.dx)*torch.softmax(-output/(self.T*self.kb), dim=1)  # 确保预测值是概率分布
        
        return output, sub_network_outputs  # Return both final output and sub-network outputs
    


# In[ ]:


# 定义Jesen熵函数
Nf=78
xmin = -3
xmax = 9
class KLDivergenceLoss(nn.Module):
    def __init__(self, reduction='batchmean',kb=0.1,dx=((xmax-xmin)/(Nf-1)),lambda_reg = 1e-4,num_networks=24):
        super(KLDivergenceLoss, self).__init__()
        self.reduction = reduction
        #self.T = nn.Parameter(torch.tensor(1.0, requires_grad=True))  # 可训练参数 T
        self.kb = kb  # Boltzmann 常数
        self.dx = dx # step size
        self.lambda_reg = lambda_reg  # Regularization weight for second derivative constraint
        self.num_networks = num_networks
    def forward(self,x, y_pred, y_true,model):
        T = x[:, 1].unsqueeze(1)  # Shape: [batch_size, 1]
        y_pred = (1/self.dx)*torch.softmax(-y_pred/(T*self.kb), dim=0) # 确保预测值是概率分布
        y_pred = torch.clamp(y_pred, min=1e-9, max=100)  # 避免 log(0)
        y_true = torch.clamp(y_true, min=1e-9, max=100)  # 避免 log(0)
        kl_pt = torch.sum(y_true * torch.log(y_true / (0.5*(y_pred+y_true))), dim=0)  # D_KL(P || Q)
        kl_tp = torch.sum(y_pred * torch.log(y_pred / (0.5*(y_pred+y_true))), dim=0)  # D_KL(Q || P)
        kl_loss = torch.mean(0.5 * (kl_pt + kl_tp)) if self.reduction == 'mean' else torch.sum(0.5 * (kl_pt + kl_tp))
        # Compute the second derivative of value w.r.t. x
        x.requires_grad_(True)  # Enable differentiation for x
        second_derivative_penalty = 0
        # Get model output (value)
        _, sub_network_outputs = model(x)  # Get sub-network outputs
        for i in range(self.num_networks // 2):
            net1_out = sub_network_outputs[:, 2*i]  # net1_out
            net2_out = sub_network_outputs[:, 2*i+1] ** 2  # Ensure net2_out is non-negative

            # Compute `value = net1_out - T * net2_out`
            value = net1_out - T * net2_out

            # Compute first derivative ∂(value)/∂x
            first_derivative = torch.autograd.grad(
                outputs=value,
                inputs=x,
                grad_outputs=torch.ones_like(value),
                create_graph=True
            )[0][:, 0]  # Select derivative w.r.t. x only

            # Compute second derivative ∂²(value)/∂x²
            second_derivative = torch.autograd.grad(
                outputs=first_derivative,
                inputs=x,
                grad_outputs=torch.ones_like(first_derivative),
                create_graph=True
            )[0][:, 0]  # Select second derivative w.r.t. x only

            # Penalize points where ∂²(value)/∂x² ≤ 0
            second_derivative_penalty = second_derivative_penalty + torch.mean(torch.clamp(-second_derivative + 1e-8, min=0.0))

        # Final loss = KL loss + constraint penalty
        total_loss = kl_loss + self.lambda_reg * second_derivative_penalty

        return total_loss




# In[ ]:


# load data

import pandas as pd
import torch


# Load CSV file as a Pandas DataFrame
df = pd.read_csv("V_T.csv",header = None)

# Convert DataFrame to PyTorch tensor (float type)
X_T_train = torch.tensor(df.values, dtype=torch.float64).to(device)

print("X_T_train shape:", X_T_train.shape)
print(X_T_train)

df = pd.read_csv("PP.csv",header = None)

# Convert DataFrame to PyTorch tensor (float type)
y_train = torch.tensor(df.values, dtype=torch.float64).to(device)
print("y_train shape:", y_train.shape)
print(y_train[25:39])


# In[ ]:


# 创建耦合模型
width = 8  # Two_NN 隐藏层宽度
model = CoupledModel(width).to(device)

# 创建优化器
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 定义损失函数
criterion = KLDivergenceLoss()

 #训练循环
num_epochs = 30001
for epoch in range(num_epochs):
    optimizer.zero_grad()  # 清空梯度
    y_pred,sub_outputs = model(X_T_train)  # 前向传播
    loss = criterion(X_T_train,y_pred, y_train,model)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新参数
    
    if epoch % 100 == 0:
       print(f"Epoch [{epoch}/{num_epochs}] - Loss: {loss.item():.10f}")


# In[ ]:


kb=0.1
xmin = -3
xmax = 9
Nf =78 
model = model.to(device)
# X_train = torch.linspace(xmin, xmax, Nf).unsqueeze(1)  # Shape: [100, 1]
# # Create a `100x1` tensor of T values
# T_tensor = torch.full((Nf, 1), T, dtype=torch.float32)
# # Concatenate X_train and T to create a `100x2` tensor
# X_T_train = torch.cat([X_train, T_tensor], dim=1)  # Shape: [100, 2]
# # dx=((xmax-xmin))/(Nf-1)
# # y_pred = model(X_T_train)
# T = X_T_train[:, 1].unsqueeze(1)
# y_pred = (1/dx)*torch.softmax(-y_pred/(T*kb), dim=0)
# y_pred = y_pred.detach().cpu().numpy()

# # Save to text file
# np.savetxt("CZ_y_pred_Fe3pt.txt", y_pred, fmt="%.10f")
print("X_T_train shape:", X_T_train.shape)

dx=((xmax-xmin))/(Nf-1)
y_pred,sub_outputs = model(X_T_train)
y_pred = y_pred.detach().cpu().numpy()
sub_outputs = sub_outputs.detach().cpu().numpy()
np.savetxt("CZ_y_pred_Fe3pt_Free_energy.txt", y_pred, fmt="%.10f")
np.savetxt("CZ_y_pred_Fe3pt_configuration.txt", sub_outputs, fmt="%.10f")

X_train = torch.linspace(xmin, xmax, Nf).unsqueeze(1).to(device)  # Shape: [100, 1]
# Create a `100x1` tensor of T values
T_tensor = torch.full((Nf, 1), 0.9664, dtype=torch.float64).to(device)
# Concatenate X_train and T to create a `100x2` tensor
X_T_train = torch.cat([X_train, T_tensor], dim=1)  # Shape: [100, 2]
y_pred,sub_outputs = model(X_T_train)
y_pred = y_pred.detach().cpu().numpy()
sub_outputs = sub_outputs.detach().cpu().numpy()
np.savetxt("CZ_y_pred_Fe3pt_Free_energy_0.txt", y_pred, fmt="%.10f")
np.savetxt("CZ_y_pred_Fe3pt_configuration_0.txt", sub_outputs, fmt="%.10f")

# Generate values
Nf = 201
NT = 596
V = torch.linspace(xmin, xmax, Nf).reshape(-1, 1)  # x values
T_vals = torch.linspace(1.0, 5.0, NT).reshape(-1, 1)  # Temperature values

# Compute Gradient for Free Energy
gradients_list = []
second_derivatives_list = []

print("First Temperature Value (T_vals[0]):", T_vals[0].item())  # Print first temperature value

for i in range(NT):  # Iterate over one temperature value
    T_scalar = T_vals[i].item()  # Convert T_vals[i] from tensor to scalar
    T_tensor = torch.full((Nf, 1), T_scalar, dtype=torch.float64).to(device)  # Expand T value for all x

    X_T_train = torch.cat([V.to(device), T_tensor], dim=1).to(device)

    X_T_train.requires_grad = True  # Enable differentiation before model call

    y_pred,sub_outputs = model(X_T_train)
    
    # Compute first-order gradient (∂y/∂x)
    first_derivative = torch.autograd.grad(
        outputs=y_pred,
        inputs=X_T_train,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True
    )[0][:, 0]  # Select derivative w.r.t. x only

    gradients_list.append(first_derivative.detach().cpu().numpy())  # Store first derivative

    # Compute second-order gradient (∂²y/∂x²)
    second_derivative = torch.autograd.grad(
        outputs=first_derivative,
        inputs=X_T_train,
        grad_outputs=torch.ones_like(first_derivative),
        create_graph=True
    )[0][:, 0]  # Select second derivative w.r.t. x only

    second_derivatives_list.append(second_derivative.detach().cpu().numpy())  # Store second derivative


# Convert to NumPy array and Transpose
first_derivatives_array = np.array(gradients_list).T  # Transpose
second_derivatives_array = np.array(second_derivatives_list).T  # Transpose

# Print computed second derivatives
print("Computed Second Derivatives (Transposed):", second_derivatives_array.shape)

# Save Transposed First and Second Derivatives to File
np.savetxt("CZ_y_pred_Fe3pt_gradient.txt", first_derivatives_array, fmt="%.10f")
np.savetxt("CZ_y_pred_Fe3pt_Hessian.txt", second_derivatives_array, fmt="%.10f")
print("First and Second derivatives saved to text files!")

# dx = (xmax-xmin)/Nf

# y_pred = (1/dx)*torch.softmax(-y_pred/(kb*T), dim=0)
# y = y_pred
# print(X_T_train.shape,y.shape)
# x = X_T_train[:,:-1]
# # Plot the solutions
# fig, ax1 = plt.subplots()
# ax1.plot(x.cpu().detach().numpy(), y.cpu().detach().numpy(), color='blue', label='Zentropy')  # Convert back to NumPy for plotting
# x_vals = torch.linspace(-2, 2, 100)
# # Compute potential values
# V_vals = V(x_vals,T)
# P_vals = boltzmann_probability(V_vals,x_vals, T)
# P_vals = P_vals.detach().cpu().numpy()
# plt.plot(x_vals, P_vals, 'r--', lw=2, label="Theoretical Distribution")  # 理论曲线

# ax1.set_xlabel('x', color='black')
# ax1.set_ylabel('f(x)', color='black')
# ax1.tick_params(axis='y', color='black')
# ax1.legend(loc='upper left')
# plt.show()

