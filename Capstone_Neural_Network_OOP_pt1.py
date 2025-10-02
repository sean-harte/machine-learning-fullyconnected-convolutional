#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 15:04:47 2024

@author: seanharte
"""

'''
small NN N ->  N//3 -> N//3 -> 1
loss function just MSE 
'''

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import seaborn as sns

plt.rcParams.update({'font.size': 14})  # Set the global font size to 10. matches overleafs default

N = 96 # number of datapoints in rho from dataset
x0i = 0.4 # starting x0 from dataset loop. used to get the 3d plot for error vs v0 and sig
dataset = np.load('dataset_no_pots.npy')
dataset = torch.from_numpy(dataset).float()

class MyDataset(Dataset):
    def __init__(self, dataset):
        self.v0 = dataset[:,0]
        self.x0 = dataset[:,1]
        self.sig = dataset[:,2]
        self.Trho = dataset[:,3]
        self.Erho = dataset[:,4]
        self.rho = dataset[:,5:5+N]
        
    def __len__(self):
        return len(self.v0)
    
    def __getitem__(self,idx):
        return self.rho[idx], self.Trho[idx], self.v0[idx], self.sig[idx], self.x0[idx]

# need to create DataLoader
nn_dataset = MyDataset(dataset) # create an instance of the class so that it can be used
train_size = int(0.8*len(nn_dataset))
test_size = len(nn_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(nn_dataset, [train_size,test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# define the class for the neural network
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet,self).__init__()
        self.fc1 = nn.Linear(N, N//3) # // is integer division
        self.fc2 = nn.Linear(N//3, N//3)
        self.fc3 = nn.Linear(N//3, 1)
        self.elu = nn.ELU()
        
    def forward(self, x):
        x = self.elu(self.fc1(x))
        x = self.elu(self.fc2(x))
        x = self.fc3(x)
        return x

# we have the class for the neural network so we need to initialise it 
# we also need to define our error functon thing, the mean squared part
# we also need to define our optimiser, use the adam one, has stocastic method in it, finds lower minimum generally
model = NeuralNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            

# training the model
n_iters = 100
lowest = 100 ## to find he lowest and highest kinetic energy for the report/seminar
highest = 0
for epoch in range(n_iters):
    model.train()
    for inputs, target, v0s, sigs, x0s in train_loader:
        
        # to find the highest and lowest kinetic energy
        for element in target:
            if element < lowest:
                lowest = element
            if element > highest:
                highest = element
        
        optimizer.zero_grad()
        outputs = model(inputs)
        # print(f'outputs = {outputs}')
        
        # loss = torch.sqrt(criterion(outputs.squeeze(),target))
        loss = criterion(outputs.squeeze(),target)
        
        # print(f'loss = {loss}')
        # if(epoch == 1) :
        #     print(loss.item(), len(train_loader))
        loss.backward() # what exactly is happening in these steps. is backward telling the optimizer what to do?
        optimizer.step() #
        
    if(epoch+1)%10==0:
        model.eval()
        with torch.no_grad():
            # test_loss = sum(torch.sqrt(criterion(model(inputs).squeeze(), target)) for inputs,target,v0s,sigs,x0s in test_loader)/len(test_loader)
            test_loss = sum(criterion(model(inputs).squeeze(), target) for inputs,target,v0s,sigs,x0s in test_loader)/len(test_loader)
        print(f'Epoch[{epoch+1}/{n_iters}], Test Loss = {test_loss.item():.4f}')


print(f'highest = {highest}')
print(f'Lowest = {lowest}')



#### PLOTTING ####
# ==========================================================================================================================================================
# ==========================================================================================================================================================
# ==========================================================================================================================================================

# PLOTTING SURFACE FIRST

# to standardise the plotting I am defining the dimensions for a regular figure
l = 9 # length
h = 6 # height

# getting the 3d graph of v0(x),sig(y),loss(z)
# output will be an array of arrays that hold values [v0,sig,loss] in that order
# all these datapoints are for x0i = 0.4 as once we have the graph proving the 
# we are plotting the absolute error between the output and the known value in percentage
# 
plot3d = []
for inputs,target,v0s,sigs,x0s in test_loader:
    for i in range(len(x0s)):
        # if(x0s[i] == x0i):
        # print(f'comp target = {model(inputs).squeeze()[i]}')
        # print(f'calc target = {target[i]}')
        # a = criterion(model(inputs).squeeze()[i],target[i])
        # a = torch.sqrt(a)
        a = abs(model(inputs).squeeze()[i] - target[i])
        a = (a/target[i])*100
        b = [v0s[i].item(),sigs[i].item(),a.item(),target[i].item()]
        plot3d.append(b)
        # print(f'\ntarget = {target[i]},\nx0s[i] = {x0s[i]},\nv0s[i] = {v0s[i]},\nsigs[i] = {sigs[i]},\nloss = {a}\n')
plot3d = torch.FloatTensor(plot3d) # convert back to tensor


# setting up the 3d surface plot
y,x,z,t = plot3d[:,0], plot3d[:,1], plot3d[:,2], plot3d[:,3]

# ==========================================================================================================================================================
# PLOTTING 3D SCATTER PLOT

#plotting relationship between v0, sig and the kinetic energy
fig0 = plt.figure(figsize=(l,h))
ax0 = fig0.add_subplot(111, projection='3d')
ax0.scatter(x,y,t, linewidth=0, antialiased=False)
ax0.set_ylabel('$V_0$')
ax0.set_xlabel('$\sigma$')
ax0.set_zlabel(r'$T[\rho]$', rotation=90)
plt.tight_layout()
plt.savefig('kin_energy_relations_in_9_6.pdf')
plt.show()

# plotting as scatter
fig1 = plt.figure(figsize=(l,h))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(x,y,z, linewidth=0, antialiased=False)
ax1.set_ylabel('$V_0$')
ax1.set_xlabel('$\sigma$')
ax1.set_zlabel('Loss (%)', rotation=90)
plt.tight_layout()
plt.savefig('scatter_in_9_6.pdf')
plt.show()

# ==========================================================================================================================================================
# PLOTTING 3D SURFACE

# plotting using interpolation to hopefully avoid the spikes in the surface plot
x_grid, y_grid = np.meshgrid(np.linspace(min(x), max(x), 25), np.linspace(min(y), max(y), 25)) # grid of data where i want to interpolate
z_grid = griddata((x,y),z, (x_grid,y_grid), method='linear') # interpolating the data

fig3 = plt.figure(figsize=(l,h))
ax3 = fig3.add_subplot(111, projection='3d')
surf3 = ax3.plot_surface(x_grid, y_grid, z_grid, cmap='viridis', linewidth=0, antialiased=False)
fig3.colorbar(surf3, ax=ax3)
ax3.set_ylabel('$V_0$')
ax3.set_xlabel('$\sigma$')
ax3.set_zlabel('Loss (%)', rotation=90) 
# ax3.set_zticklabels([])
plt.tight_layout()
plt.savefig('surface_in_9_6.pdf')
plt.show()


# graph below is for the graph above but fit using a cubic interpolation method.
# =============================================================================
# # plotting using interpolation to hopefully avoid the spikes in the surface plot
# x_grid, y_grid = np.meshgrid(np.linspace(min(x), max(x), 25), np.linspace(min(y), max(y), 25)) # grid of data where i want to interpolate
# z_grid = griddata((x,y),z, (x_grid,y_grid), method='cubic') # interpolating the data
# fig4 = plt.figure()
# ax4 = fig4.add_subplot(111, projection='3d')
# surf4 = ax4.plot_surface(x_grid, y_grid, z_grid, cmap='viridis', linewidth=0, antialiased=False)
# fig4.colorbar(surf4, ax=ax4)
# ax4.set_ylabel('$V_0$')
# ax4.set_xlabel('$\sigma$')
# ax4.set_zlabel('Loss')
# plt.show()
# =============================================================================



# ==========================================================================================================================================================
# ==========================================================================================================================================================
# ==========================================================================================================================================================

# PLOTTING TRANSLATIONAL INVARIANCE 

# now I want the plot that wil show the translational invariance
# need to have randomly shuffled v0 and sig for x0 in increasing order
# plot error as a percentage and the error is not MSE it is absolute
x0i = 0.4
x0f = 0.6
x0_n = 15
x0_array = np.linspace(x0i,x0f,x0_n)
tran_inv_plot = []
for x0 in x0_array:
    for inputs,target,v0s,sigs,x0s in test_loader:
        for i in range(len(x0s)):
            if(x0s[i] == x0):
                # a = criterion(model(inputs).squeeze()[i],target[i])
                a = abs(model(inputs).squeeze()[i] - target[i])
                a = (a/target[i])*100
                b = [x0s[i].item(),a.item()] # all i need is x0 and loss. i could actually do the average loss for each x0 and that would get rid of the grouping on the graph issue
                tran_inv_plot.append(b)
                # print(f'\nx0s[i] = {x0s[i]}, loss = {a}\n')
tran_inv_plot = torch.FloatTensor(tran_inv_plot) # convert back to tensor


x_col = tran_inv_plot[:,0]
for i in range(len(x_col)): # this for loop is just rounding every element in the x_col
    x_col[i] = round(x_col[i].item(),2)
x_test = torch.arange(1,len(x_col)+1,1)
y_col = tran_inv_plot[:,1]

# ==========================================================================================================================================================

# PLOTTING SWARMPLOT

## plotting with swarmplot. you lose some data if the markers are too big which they need to be in order to see them
fig6 = plt.figure(figsize = (2*l,h))
ax6 = sns.swarmplot(x=x_col,y=y_col, size=5)
ax6.set_ylabel('Loss (%)')
ax6.set_xlabel('$x_0$')
ax6.grid()
plt.tight_layout()
plt.savefig('swarm_in_18_6.pdf')
plt.show()


# ## plotting normally
# fig7 = plt.figure(figsize = (2*l,h))
# ax7 = fig7.add_subplot(111)
# ax7.plot(x_col,y_col)
# ax7.set_ylabel('Loss (%)')
# ax7.set_xlabel('$x_0$')
# ax7.grid()
# plt.tight_layout()
# plt.savefig('line_in_18_6.pdf')
# plt.show()

# ==========================================================================================================================================================

# PLOTTING X0 ERROR LINEARLY

## plotting linearly for every element

def draw_brace(ax, xspan, yy, text): # function for drawing labelled braces on a plot

    xmin, xmax = xspan
    xspan = xmax - xmin
    ax_xmin, ax_xmax = ax.get_xlim()
    xax_span = ax_xmax - ax_xmin

    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin
    resolution = int(xspan/xax_span*100)*2+1 # guaranteed uneven
    beta = 300./xax_span # the higher this is, the smaller the radius

    x = np.linspace(xmin, xmax, resolution)
    x_half = x[:int(resolution/2)+1]
    y_half_brace = (1/(1.+np.exp(-beta*(x_half-x_half[0])))
                    + 1/(1.+np.exp(-beta*(x_half-x_half[-1]))))
    y = np.concatenate((y_half_brace, y_half_brace[-2::-1]))
    y = yy + (.05*y - .01)*yspan # adjust vertical position

    ax.autoscale(False)
    ax.plot(x, y, color='black', lw=1)

    ax.text((xmax+xmin)/2., yy+.07*yspan, text, ha='center', va='bottom')

fig8 = plt.figure(figsize = (2*l,h))
ax8 = fig8.add_subplot(111)
ax8.plot(x_test,y_col)
ax8.set_ylabel('Loss (%)')
ax8.set_xlabel('$x_0$')
ax8.set_xticklabels([])

x_unique = torch.unique(x_col)

for i in range(len(x_unique)): # this is to draw the brackets
    if(i==0):
        draw_brace(ax8, ((900/15)*i, (900/15)*(i+1)), max(y_col)*0.85, f'$x_0$={x_unique[i]:.2f}')
    else:
        draw_brace(ax8, ((900/15)*i, (900/15)*(i+1)), max(y_col)*0.85, f'{x_unique[i]:.2f}')

ax8.grid()
plt.tight_layout()
plt.savefig('spreadoutx_in_18_6.pdf')
plt.show()

# ==========================================================================================================================================================

# PLOTTING AVERAGED X0 ERROR

## plotting averaged error for each value of x0. lose the information about outliers but can plot it better
index_list = []
for element in x_unique: # this is to get the no. of specific x0 which I will use for indexing purposes later
      a = 0
      for i in range(len(x_col)):
          if (element == x_col[i]):
              a+=1
      index_list.append(a)
index_list = torch.FloatTensor(index_list)

avg_loss = []
for i in range(len(index_list)):
    av_l = sum(y_col[int(sum(index_list[:i])):int(sum(index_list[:i+1]) - 1)])/index_list[i]  # genuinely so sorry for this line it is probably unreadable. I am essentially summing the number of points that correspond to each value of x0, I found the number of times each x0 appears in index_list
    avg_loss.append(av_l)
avg_loss = torch.FloatTensor(avg_loss)

fig9 = plt.figure(figsize=(l,h/2))
ax9 = fig9.add_subplot(111)
ax9.plot(x_unique, avg_loss)
ax9.set_ylabel('Average Loss (%)')
ax9.set_xlabel('$x_0$')
ax9.grid()
plt.tight_layout()
plt.savefig('avgtrainv_in_18_6.pdf')
plt.show()

# ==========================================================================================================================================================
# ==========================================================================================================================================================
# ==========================================================================================================================================================

# =============================================================================
# # PLOTTING TRAIN-TEST CURVE
# 
# # create another instance of the class to get an untrained model. call it model1
# model1 = NeuralNet()
# optimizer = torch.optim.Adam(model1.parameters(), lr=0.01)
# 
# # training the model1
# n_iters1 = 150
# train_loss_array = torch.zeros(n_iters1)
# test_loss_array = torch.zeros(n_iters1)
# 
# for epoch in range(n_iters1):
#     # getting train loss values
#     model1.train()
#     for inputs, target, v0s, sigs, x0s in train_loader:
#         optimizer.zero_grad()
#         outputs = model1(inputs)
#         loss = criterion(outputs.squeeze(),target) 
#         # print(f'Epoch = {epoch+1},train loss= {loss}')
#         loss.backward() # what exactly is happening in these steps. is backward telling the optimizer what to do?
#         optimizer.step() #
#         with torch.no_grad():
#             train_loss_array[epoch] += loss/len(train_loader)
#     
#     # getting test loss values
#     model1.eval()
#     for inputs1, target1, v0s1, sigs1, x0s1 in test_loader:
#         test_loss = criterion(model1(inputs1).squeeze(), target1) 
#         with torch.no_grad():
#             # print(f'Epoch[{epoch+1}/{n_iters1}], Test Loss = {test_loss.item():.4f}')
#             test_loss_array[epoch] += test_loss/len(test_loader)
#     
#     if(epoch+1)%10==0:
#         print(f'Epoch[{epoch+1}/{n_iters1}]')
# 
# # train_loss_array = train_loss_array/len(train_loader)
# # test_loss_array = test_loss_array/len(test_loader)
# epoch_array = torch.arange(1,n_iters1+1)
# val = 0
# 
# fig10 = plt.figure(figsize=(l,h/2))
# ax10 = fig10.add_subplot(111)
# ax10.plot(epoch_array[val:], train_loss_array[val:], label = 'Train Loss',linewidth=2)
# ax10.plot(epoch_array[val:], test_loss_array[val:], label = 'Test Loss',linewidth=2)
# ax10.set_ylabel('Loss')
# ax10.set_xlabel('Epoch')
# ax10.grid()
# ax10.legend()
# plt.tight_layout()
# plt.savefig('train_test_curve1_in_9_6.pdf')
# plt.show()
# =============================================================================

# ==========================================================================================================================================================
# ==========================================================================================================================================================
# ==========================================================================================================================================================

# PLOTTING PARITY PLOT

# need train_targets, train_calculated and test_targets, test_calculated
train_targets = torch.tensor([target for inputs,target,v0s,sigs,x0s in train_dataset])
train_calculated = torch.tensor([model(inputs) for inputs,target,v0s,sigs,x0s in train_dataset])

test_targets = torch.tensor([target for inputs,target,v0s,sigs,x0s in test_dataset])
test_calculated = torch.tensor([model(inputs) for inputs,target,v0s,sigs,x0s in test_dataset])

rmse = sum((test_targets-test_calculated)**2)/len(test_targets)

# for the straight line
n = 100
straight_x = np.linspace(min(train_targets)-1, max(train_targets)+1, n)
straight_y = np.linspace(min(train_targets)-1, max(train_targets)+1, n)

fig11 = plt.figure(figsize=(l,h))
ax11 = fig11.add_subplot(111)
ax11.plot(straight_x,straight_y, color='black')
ax11.scatter(train_calculated, train_targets, label = 'Train', s=3)
ax11.scatter(test_calculated, test_targets, label = 'Test', s=3)
ax11.plot([], [], ' ', label=r"RMS_test = {rmse:.2f}")
ax11.set_ylabel('vW')
ax11.set_xlabel('ML')
ax11.grid()
ax11.legend(markerscale=3)
plt.tight_layout()
plt.savefig('parity_plot_in_9_6.pdf')
plt.show()


# ==========================================================================================================================================================
# ==========================================================================================================================================================
# ==========================================================================================================================================================

# PLOTTING MOSAIC PLOT OF GRAPHS

# matplotlib mosaic plot with both of them on it now
layout = [['A','D','D','D'],
          ['A','E','E','E'],
          ['B','B','C','C']
          ]

fig12, axd12 = plt.subplot_mosaic(layout, figsize=(l,h))
### for A ###
axd12['A'].plot(straight_x,straight_y, color='black')
axd12['A'].scatter(train_calculated, train_targets, label = 'Train', s=3)
axd12['A'].scatter(test_calculated, test_targets, label = 'Test', s=3)
axd12['A'].set_ylabel('vW')
axd12['A'].set_xlabel('ML')
axd12['A'].grid()
axd12['A'].legend()

### for B ###
axd12['B'].plot(epoch_array, train_loss_array, label = 'Train Loss')
axd12['B'].plot(epoch_array, test_loss_array, label = 'Test Loss')
axd12['B'].set_ylabel('Loss')
axd12['B'].set_xlabel('Epoch')
axd12['B'].grid()
axd12['B'].legend()

### for C ###
axd12['C'].plot(x_unique, avg_loss)
axd12['C'].set_ylabel('Average Loss')
axd12['C'].set_xlabel('$x_0$')
axd12['C'].grid()

### for D ###
axd12['D'].plot(x_test,y_col, linewidth=0.5)
axd12['D'].set_ylabel('Loss')
axd12['D'].set_xlabel('$x_0$')
axd12['D'].set_xticklabels([])

x_unique = torch.unique(x_col)

for i in range(len(x_unique)): # this is to draw the brackets
    draw_brace(axd12['D'], ((900/15)*i, (900/15)*(i+1)), max(y_col)*0.85, f'{x_unique[i]:.2f}') #, f'$x_0$={x_unique[i]:.2f}'

axd12['D'].grid()

### for E ###
sns.swarmplot(x=x_col,y=y_col, size=2, ax = axd12['E'])
axd12['E'].set_ylabel('Loss')
axd12['E'].set_xlabel('$x_0$')
#axd12['E'].set_xticklabels([])
axd12['E'].grid()

# appears you cant really add a 3d surface plot to the mosaic structure
# =============================================================================
# ### for F ###
# axd12['F'] = fig12.add_subplot(121, projection='3d')
# surf12 = axd12['F'].plot_surface(x_grid, y_grid, z_grid, cmap='viridis', linewidth=0, antialiased=False)
# fig12.colorbar(surf12, ax=axd12['F'])
# axd12['F'].set_ylabel('$V_0$')
# axd12['F'].set_xlabel('$\sigma$')
# axd12['F'].set_zlabel('Loss', rotation=90) 
# # axd12['F'].set_zticklabels([])
# =============================================================================

plt.tight_layout()
plt.savefig('mosaic_in_9_6.pdf')
plt.show()

# axd5['B'].plot_surface(x_grid, y_grid, z_grid, cmap='viridis', linewidth=0, antialiased=False)
# plt.tight_layout()
# plt.show()





