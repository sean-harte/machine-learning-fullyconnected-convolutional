#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:30:36 2025

@author: seanharte
"""

'''
applying 1d convolution to the system: convolution_output (CO) -> CO -> CO//3 -> 1
loss is MSE and theta
'''


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import seaborn as sns

plt.rcParams.update({'font.size': 12})  # Set the global font size to 10. matches overleafs default

N = 96 # number of datapoints in rho from dataset
x0i = 0.4 # starting x0 from dataset loop. used to get the 3d plot for error vs v0 and sig
dataset = np.load('dataset.npy')
dataset = torch.from_numpy(dataset).float()
x_array = torch.linspace(0,1,N)

batch = 32 # batch size. 16 divides evenly into the train set so that the channels of the CNN line up but doesnt divide evenly into the test set so the test set thing is commented out

conv1_in_ch = 1
conv1_out_ch = 4
kern1_size = 3

conv2_in_ch = conv1_out_ch
conv2_out_ch = 8
kern2_size = 5

maxpool_kern1 = 3
maxpool_stride1 = 2

maxpool_kern2 = maxpool_kern1
maxpool_stride2 = maxpool_stride1

group_size = 1

class MyDataset(Dataset):
    def __init__(self, dataset):
        self.v0 = dataset[:,0]
        self.x0 = dataset[:,1]
        self.sig = dataset[:,2]
        self.Trho = dataset[:,3]
        self.Erho = dataset[:,4]
        self.rho = dataset[:,5:5+N] 
        self.pot = dataset[:,5+N:5+N+N]
        
    def __len__(self):
        return len(self.v0)
    
    def __getitem__(self,idx):
        return self.rho[idx], self.Trho[idx], self.v0[idx], self.sig[idx], self.x0[idx], self.pot[idx]

# need to create DataLoader
nn_dataset = MyDataset(dataset) # create an instance of the class so that it can be used
train_size = int(0.8*len(nn_dataset))
test_size = len(nn_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(nn_dataset, [train_size,test_size])

train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False)

# define the class for the neural network
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet,self).__init__()
        self.conv1 = nn.Conv1d(conv1_in_ch, conv1_out_ch, kern1_size, groups = group_size, padding = kern1_size)  # (kern1_size//2 + 1)
        self.conv2 = nn.Conv1d(conv2_in_ch, conv2_out_ch, kern2_size, padding = kern2_size)  # (kern2_size//2 + 1)
        self.maxpool1d_1 = nn.MaxPool1d(maxpool_kern1, maxpool_stride1) # default stride is kernal size for MaxPool1d
        self.maxpool1d_2 = nn.MaxPool1d(maxpool_kern2, maxpool_stride2)
        
        # self.fc1_in = self.get_output_dim((1, conv1_in_ch, N)) # input shape doesnt care about the batch size so here ive just put it as 1
        
        # self.fc1 = nn.Linear(self.fc1_in, self.fc1_in) # // is integer division
        # self.fc2 = nn.Linear(self.fc1_in, self.fc1_in//3)
        # self.fc3 = nn.Linear(self.fc1_in//3, 1)
        self.relu = nn.ELU()
        
    def forward(self, x):
        #print(f"Input shape before conv1: {x.shape}")
        x = x.unsqueeze(1)
        # print('cool1')
        #print(f"Input shape before conv1 after unsqueeze: {x.shape}")
        x = self.relu(self.conv1(x))  # self.maxpool1d_1(
        # print('cool2')
        x = self.relu(self.conv2(x))  # self.maxpool1d_2(
        # print('cool3')
        # print(f'Before .view = {x.size()}')
        # print(x)
        # print(self.fc1_in)
        
        # x = x.view(-1,self.fc1_in)  # Flatten while keeping batch size
        # print(x, 'here1')
        # print(x.shape, 'here2')
        x = torch.sum(x,dim=1)
        # print(x, 'here3')
        # print(x.shape, 'here4')
        x = torch.sum(x,dim=1)
        # print(x, 'here5')
        # print(x.shape, 'here6')
        # x = x.sum()
        # print(x, 'here7')
        # print(f'After .view = {x.size()}')
        
        # x = self.relu(self.fc1(x))
        # print('cool4')
        # x = self.relu(self.fc2(x))
        # print('cool5')
        # x = self.fc3(x)
        # print('cool6')
        return x
    
    def get_output_dim(self, input_shape): # function to get the flattened output size of the convolutional layers so that it can be passed to the fc1 definition
        self.relu = nn.ELU()
        with torch.no_grad():
            dummy_input = torch.zeros(input_shape)
            output = dummy_input
            output = self.relu(self.conv1(output))  # self.maxpool1d_1(
            output = self.relu(self.conv2(output))  # self.maxpool1d_2( 
            flattened_size = output.numel() // output.shape[0]  # Compute size per sample
        return flattened_size
    

# we have the class for the neural network so we need to initialise it 
# we also need to define our error functon thing, the mean squared part
# we also need to define our optimiser, use the adam one, has stocastic method in it, finds lower minimum generally
model = NeuralNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# training the model
n_iters = 150
train_loss_array = torch.zeros(n_iters)
test_loss_array = torch.zeros(n_iters) ### for the train test curve later
theta_vs_epoch_array = torch.zeros(n_iters) ### average test theta per epoch vs epochs
epoch_array = torch.arange(1,n_iters+1)
v0_x0_sig_target_theta = []
v0_x0_sig_percentageloss_theta = []

funcderiv_scalar = 90

for epoch in range(n_iters):
    model.train()
    i=0
    for inputs, target, v0s, sigs, x0s, pots in train_loader:
        inputs.requires_grad_(True) ## do not put grad on the targets. absolutely dont want the system learning things form the target
        optimizer.zero_grad()
# =============================================================================
#         if inputs.grad != None: ### do these even fucking work??
#             inputs.grad.zero_()
#             print('Here')
# =============================================================================

        outputs = model(inputs)
        
        # funcderiv = torch.autograd.grad(((target - outputs)**2).sum(),inputs,create_graph=True,retain_graph=True)[0] #.requires_grad_(True)  # this should be the functional derivative of Tml[rho]
        funcderiv = funcderiv_scalar*torch.autograd.grad(outputs.sum(),inputs,create_graph=True,retain_graph=True)[0]
        
        funcderiv_plus_v = funcderiv + pots # this is the functional derivative of Tml plus the potential 
        const_vec = -torch.ones_like(funcderiv_plus_v) # this is the vector of -1's that is the same shape as the functional derivative
        
        top = (const_vec*funcderiv_plus_v).sum(dim=1)
        bottom = (torch.norm(const_vec,dim=1)*torch.norm(funcderiv_plus_v,dim=1))
        cos_theta = top/bottom # defining error withh cos(theta) insteas of theta to avoid the cos() operations. doesnt converge better as far as i can see
        
        theta = torch.arccos(top/bottom) # this is the expression for theta which gives the error between the funcderiv_plus_v and the lagrange multiplier (which is the chemical potential i think) which is a constant in the form of an angle theta. this angle theta has as many entries as terms in each inputs i.e. the batch size
        
        root_theta = torch.sqrt(theta) # testing if convergence is better if theta is square rooted and i use the square root in error
        
        theta_scalar = 10.0
        increasing_scalar = theta_scalar*((epoch+1)/n_iters)
        
        if (((epoch+1)%10==0) and (i==0)):
            print('========================================================')
            print(f'Epoch = [{epoch+1}/{n_iters}]')
            print(f'\ntheta loss = {(torch.sum(theta)/len(theta)):.4f}') 
            # print(f'\ncos(theta) loss = {(torch.sum(cos_theta)/len(cos_theta))}')
            print(f'MSE loss = {criterion(outputs.squeeze(),target):.4f}')
            print(f'Total Train Loss = {((torch.sum(theta)/len(theta)) + criterion(outputs.squeeze(),target)):.4f}')
            # print(f'theta = {theta}')
            
            plt.plot(funcderiv[0].detach().numpy())
            plt.title(f'Func Deriv: Epoch = {epoch+1}, i = {i}')
            plt.show()
            
            plt.plot(funcderiv_plus_v[0].detach().numpy())
            plt.title(f'Func Deriv + V: Epoch = {epoch+1}, i = {i}')
            plt.show()
            # print('======================================================')
            
        # sum_cos_theta = torch.sum(cos_theta)
        # sum_theta = torch.sum(theta) # not squaring the theta term leads to better error convergence. theta is 0 to pi so doesnt matter for positive definiteness
        # sum_root_theta = torch.sum(root_theta)
        # loss =  torch.abs(sum_cos_theta/len(cos_theta)) - 1 # criterion(outputs.squeeze(),target) + # this is the loss if we are taking cos(theta) = 1 as perfect. might work better than just theta. dont think it does
# =============================================================================
#         if(epoch<1):
#             loss =  theta_scalar*(torch.sum(theta))/len(theta) 
#         else:
# =============================================================================

        MSE = criterion(outputs.squeeze(),target)
        
        train_loss_array[epoch] += MSE/len(train_loader) ## for train test curve later    
        
        loss = MSE + theta_scalar*(torch.sum(theta))/len(theta)
        
        loss.backward() 
        optimizer.step() 
        i+=1
        
    model.eval()
    with torch.no_grad():
        test_loss = sum(criterion(model(inputs).squeeze(),target) for inputs,target,v0s,sigs,x0s,pots in test_loader)/len(test_loader)  ## for train test curve later
        test_loss_array[epoch] += test_loss
        
# =============================================================================
# 2 was here
# =============================================================================

# =============================================================================
# 4 was here
# =============================================================================

    model.eval()
    # with torch.no_grad():
    MSE = 0
    test_theta = 0
    for inputs,target,v0s,sigs,x0s,pots in test_loader: # theta_scalar*(sum_theta)/len(theta) + 
        ### for MSE ###
        MSE_loss = criterion(model(inputs).squeeze(),target) 
        MSE += MSE_loss/len(test_loader)
        
        ### for theta ###
        inputs.requires_grad_(True)
        outputs = model(inputs)
        # funcderiv = torch.autograd.grad(((outputs-target)**2).sum(),inputs,create_graph=True,retain_graph=True)[0] #.requires_grad_(True)  # this should be the functional derivative of Tml[rho]
        funcderiv = funcderiv_scalar*torch.autograd.grad(outputs.sum(),inputs,create_graph=True,retain_graph=True)[0] #.requires_grad_(True)  # this should be the functional derivative of Tml[rho]
        # funcderiv = funcderiv_scalar*funcderiv
        funcderiv_plus_v = funcderiv + pots # this is the functional derivative of Tml plus the potential 
        const_vec = -torch.ones_like(funcderiv_plus_v) # this is the vector of -1's that is the same shape as the functional derivative
        
        top = (const_vec*funcderiv_plus_v).sum(dim=1)
        bottom = (torch.norm(const_vec,dim=1)*torch.norm(funcderiv_plus_v,dim=1))
        cos_theta = top/bottom # defining error withh cos(theta) insteas of theta to avoid the cos() operations. doesnt converge better as far as i can see
        
        theta = torch.arccos(top/bottom) # this is the expression for theta which gives the error between the funcderiv_plus_v and the lagrange multiplier (which is the chemical potential i think) which is a constant in the form of an angle theta. this angle theta has as many entries as terms in each inputs i.e. the batch size
        theta_sum = torch.sum(theta)/len(theta)
        test_theta += theta_sum/len(test_loader)
        
        theta_vs_epoch_array[epoch] += theta_sum/len(test_loader)
        
        if(epoch == n_iters-1):
            for j in range(len(x0s)):
                a = abs(outputs.squeeze()[j] - target[j])/target[j]
                a = a*100
                # print(theta)
                b = [v0s[j].item(),x0s[j].item(),sigs[j].item(),a.item(),theta[j].item()]
                v0_x0_sig_percentageloss_theta.append(b)
                
            for j in range(len(x0s)):
                # if(x0s[j]==0.5):
                    
                figq = plt.figure(figsize = (9,6))
                axq = figq.add_subplot(111)
                axq.plot(x_array,funcderiv[j].detach(), label='ML Functional Derivative', color='royalblue',linewidth=1.5)
                axq.plot(x_array, -pots[j]+funcderiv[j,0].detach(), label='Ideal Functional Derivative', color='darkviolet',linewidth=1.5)
                axq.set_ylabel(r'$\frac{\delta T[\rho]}{\delta \rho}$')
                axq.set_xlabel(r'$\rho(x)$')
                axq.grid()
                axq.legend()
                plt.tight_layout()
                plt.savefig('ML_funcderiv_9_6_pt6.pdf')
                plt.show()
                
                # figw = plt.figure(figsize = (9,6))
                # axw= figw.add_subplot(111)
                # axw.plot(x_array, -pots[j]+funcderiv[j,0].detach(), label='Ideal Functional Derivative', color='darkviolet',linewidth=1.5)
                # axw.set_ylabel(r'$\frac{\delta T[\rho]}{\delta \rho}$')
                # axw.set_xlabel(r'$\rho(x)$')
                # axw.grid()
                # axw.legend()
                # plt.tight_layout()
                # plt.savefig('Ideal_funcderiv_9_6_pt51.pdf')
                # plt.show()
                
                break
                
            
    if((epoch+1)%10==0):    
        print('\n### TEST ###')
        print(f'MSE Loss (just MSE) = {MSE:.4f}')
        print(f'Theta Test Loss = {test_theta:.4f}')
        print('=============================================================================')


v0_x0_sig_percentageloss_theta = torch.FloatTensor(v0_x0_sig_percentageloss_theta) # convert back to tensor


width = 2
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
# translational invariance then it doesnt matter which x0 we choose but its better 
# that we are keeping the x0 consistent
plot3d = []
for inputs,target,v0s,sigs,x0s,pots in test_loader: ########### need inputs with grad. and then import the whole theta definition
    for i in range(len(x0s)):
        # if(x0s[i] == x0i):
            outs = model(inputs)
            # a = criterion(outs.squeeze()[i],target[i])
            a = abs(outs.squeeze()[i] - target[i])/target[i]
            # print(outs.squeeze()[i] - target[i])
            # print(a.item())
            a = a*100
            # print(a.item())
            b = [v0s[i].item(),sigs[i].item(),a.item()]
            plot3d.append(b)
            # print(f'\ntarget = {target[i]},\nx0s[i] = {x0s[i]},\nv0s[i] = {v0s[i]},\nsigs[i] = {sigs[i]},\nloss = {a}\n')
plot3d = torch.FloatTensor(plot3d) # convert back to tensor


# setting up the 3d surface plot
y,x,z = plot3d[:,0], plot3d[:,1], plot3d[:,2]

# ==========================================================================================================================================================
# PLOTTING 3D SCATTER PLOT

# plotting as scatter
fig1 = plt.figure(figsize=(l,h))
ax1 = fig1.add_subplot(111, projection='3d')
ax1.scatter(x,y,z, linewidth=0, antialiased=False)
ax1.set_ylabel('$V_0$')
ax1.set_xlabel('$\sigma$')
ax1.set_zlabel('Loss (%)', rotation=90)
plt.tight_layout()
plt.savefig('scatter_in_9_6_pt6.pdf')
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
plt.savefig('surface_in_9_6_pt6.pdf')
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
x0i = 0.4
x0f = 0.6
x0_n = 15
x0_array = np.linspace(x0i,x0f,x0_n)
tran_inv_plot = []
for x0 in x0_array:
    for inputs,target,v0s,sigs,x0s,pots in test_loader:
        for i in range(len(x0s)):
            if(x0s[i] == x0):
                # a = criterion(model(inputs).squeeze()[i],target[i])
                a = abs(model(inputs).squeeze()[i] - target[i])/target[i]
                a = a*100
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
plt.savefig('swarm_in_18_6_pt6.pdf')
plt.show()


## plotting normally
fig7 = plt.figure(figsize = (2*l,h))
ax7 = fig7.add_subplot(111)
ax7.plot(x_col,y_col)
ax7.set_ylabel('Loss (%)')
ax7.set_xlabel('$x_0$')
ax7.grid()
plt.tight_layout()
plt.savefig('line_in_18_6_pt6.pdf')
plt.show()

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
    draw_brace(ax8, ((900/15)*i, (900/15)*(i+1)), max(y_col)*0.85, f'$x_0$={x_unique[i]:.2f}')

ax8.grid()
plt.tight_layout()
plt.savefig('spreadoutx_in_18_6_pt6.pdf')
plt.show()

# ==========================================================================================================================================================

# PLOTTING AVERAGED X0 ERROR

## plotting averaged error for each value of x0. lose the information about outliers but can plot it better
index_list = []
for element in x_unique: # this is to get the no of specific x0 which I will use for indexing purposes later
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

fig9 = plt.figure(figsize=(l,h))
ax9 = fig9.add_subplot(111)
ax9.plot(x_unique, avg_loss)
ax9.set_ylabel('Average Loss (%)')
ax9.set_xlabel('$x_0$')
ax9.grid()
plt.tight_layout()
plt.savefig('avgtrainv_in_9_6_pt6.pdf')
plt.show()

# ==========================================================================================================================================================
# ==========================================================================================================================================================
# ==========================================================================================================================================================

# PLOTTING TRAIN-TEST CURVE

# create another instance of the class to get an untrained model. call it model1
model1 = NeuralNet()
optimizer = torch.optim.Adam(model1.parameters(), lr=0.01)

# training the model1
# n_iters1 = 100
# train_loss_array = torch.zeros(n_iters1)
# test_loss_array = torch.zeros(n_iters1)

# =============================================================================
# for epoch in range(n_iters1):
#     # getting train loss values
#     model1.train()
#     for inputs, target, v0s, sigs, x0s, pots in train_loader:
#         inputs.requires_grad_(True) ## do not put grad on the targets. absolutely dont want the system learning things form the target
#         optimizer.zero_grad()
#         if inputs.grad != None: ### do these even fucking work??
#             inputs.grad.zero_()
#             # print(f'no.1 i={i}, epoch={epoch}')
#         outputs = model1(inputs)
#         # outputs1 = model(inputs)
#         
#         funcderiv = torch.autograd.grad(outputs.sum(),inputs, retain_graph=True)[0]  # this should be the functional derivative of Tml[rho]
#         # ^ this is going to have to go into the main block as we are redefining the loss entirely. we want the system to be trained in a different way
#         
#         funcderiv_plus_v = funcderiv + pots # this is the functional derivative of Tml plus the potential 
#         const_vec = -torch.ones_like(funcderiv_plus_v) # this is the vector of -1's that is the same shape as the functional derivative
#         
#         top = (const_vec*funcderiv_plus_v).sum(dim=1)
#         bottom = (torch.norm(const_vec,dim=1)*torch.norm(funcderiv_plus_v,dim=1))
#         theta = torch.arccos(top/bottom) # this is the expression for theta which gives the error between the funcderiv_plus_v and the lagrange multiplier (which is the chemical potential i think) which is a constant in the form of an angle theta. this angle theta has as many entries as terms in each inputs i.e. the batch size
#         
#         # theta_scalar = 100.0
#         
#         loss = criterion(outputs.squeeze(),target) + theta_scalar*(theta.sum()/len(theta))
#         loss.backward() # what exactly is happening in these steps. is backward telling the optimizer what to do?
#         optimizer.step() #
#         
#         with torch.no_grad():
#             train_loss_array[epoch] += loss/len(train_loader)
#         
#         
#         # getting test loss values
#         model1.eval()
#         for inputs1, target1, v0s1, sigs1, x0s1, pots1 in test_loader:
#             test_loss = criterion(model1(inputs1).squeeze(), target1) + theta_scalar*(theta.sum()/len(theta))
#             with torch.no_grad():
#                 # print(f'Epoch[{epoch+1}/{n_iters1}], Test Loss = {test_loss.item():.4f}')
#                 test_loss_array[epoch] += test_loss/len(test_loader)
#         
#     if(epoch+1)%10==0:
#         print(f'Epoch[{epoch+1}/{n_iters1}], Test Loss = {test_loss.item():.4f}')
# =============================================================================

# train_loss_array = train_loss_array/len(train_loader)
# test_loss_array = test_loss_array/len(test_loader)
# epoch_array = torch.arange(1,n_iters1+1)

train_loss_array = train_loss_array.detach()
test_loss_array = test_loss_array.detach()
epoch_array = torch.arange(1,n_iters+1)

fig10 = plt.figure(figsize=(l,h))
ax10 = fig10.add_subplot(111)
ax10.plot(epoch_array, train_loss_array, label = 'Train Loss')
ax10.plot(epoch_array, test_loss_array, label = 'Test Loss')
ax10.set_ylabel('Loss (%)')
ax10.set_xlabel('Epoch')
ax10.grid()
ax10.legend()
plt.tight_layout()
plt.savefig('train_test_curve1_in_9_6_pt6.pdf')
plt.show()

# ==========================================================================================================================================================
# ==========================================================================================================================================================
# ==========================================================================================================================================================

# PLOTTING PARITY PLOT

# need train_targets, train_calculated and test_targets, test_calculated
# train_targets = torch.tensor([target for inputs,target,v0s,sigs,x0s,pots in train_dataset])
# train_calculated = torch.tensor([model(inputs) for inputs,target,v0s,sigs,x0s,pots in train_dataset])

# test_targets = torch.tensor([target for inputs,target,v0s,sigs,x0s,pots in test_dataset])
# test_calculated = torch.tensor([model(inputs) for inputs,target,v0s,sigs,x0s,pots in test_dataset])

with torch.no_grad():
    train_targets = []
    train_calculated = []
    for inputs,target,v0s,sigs,x0s,pots in train_loader:
        for element in target:
            train_targets.append(element)
        
        for item in model(inputs):
            train_calculated.append(item.item())
    train_targets = torch.FloatTensor(train_targets)
    train_calculated = torch.FloatTensor(train_calculated)

test_targets = []
test_calculated = []
for inputs,target,v0s,sigs,x0s,pots in test_loader:
    for element in target:
        test_targets.append(element)
    
    for item in model(inputs):
        test_calculated.append(item.item())
test_targets = torch.FloatTensor(test_targets)
test_calculated = torch.FloatTensor(test_calculated)


# for the straight line
n = 100
straight_x = np.linspace(min(train_targets)-1, max(train_targets)+1, n)
straight_y = np.linspace(min(train_targets)-1, max(train_targets)+1, n)

fig11 = plt.figure(figsize=(l,h))
ax11 = fig11.add_subplot(111)
ax11.plot(straight_x,straight_y, color='black')
ax11.scatter(train_calculated, train_targets, label = 'Train', s=3)
ax11.scatter(test_calculated, test_targets, label = 'Test', s=3)
ax11.set_ylabel('vW')
ax11.set_xlabel('ML')
ax11.grid()
ax11.legend(markerscale=3)
plt.tight_layout()
plt.savefig('parity_plot_in_9_6_pt6.pdf')
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
axd12['B'].set_ylabel('Loss (%)')
axd12['B'].set_xlabel('Epoch')
axd12['B'].grid()
axd12['B'].legend()

### for C ###
axd12['C'].plot(x_unique, avg_loss)
axd12['C'].set_ylabel('Average Loss (%)')
axd12['C'].set_xlabel('$x_0$')
axd12['C'].grid()

### for D ###
axd12['D'].plot(x_test,y_col, linewidth=0.5)
axd12['D'].set_ylabel('Loss (%)')
axd12['D'].set_xlabel('$x_0$')
axd12['D'].set_xticklabels([])

x_unique = torch.unique(x_col)

for i in range(len(x_unique)): # this is to draw the brackets
    draw_brace(axd12['D'], ((900/15)*i, (900/15)*(i+1)), max(y_col)*0.85, f'{x_unique[i]:.2f}') #, f'$x_0$={x_unique[i]:.2f}'

axd12['D'].grid()

### for E ###
sns.swarmplot(x=x_col,y=y_col, size=2, ax = axd12['E'])
axd12['E'].set_ylabel('Loss (%)')
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
plt.savefig('mosaic_in_9_6_pt6.pdf')
plt.show()

# axd5['B'].plot_surface(x_grid, y_grid, z_grid, cmap='viridis', linewidth=0, antialiased=False)
# plt.tight_layout()
# plt.show()







####################################
########   THETA PLOTTING ##########
####################################

### THETA VS EPOCHS PLOT ### 

fig13 = plt.figure(figsize=(l,h))
ax13 = fig13.add_subplot(111)
ax13.plot(epoch_array,theta_vs_epoch_array.detach(), color='darkviolet',linewidth=width)
ax13.set_ylabel(r'$\theta$')
ax13.set_xlabel('Epoch')
ax13.grid()
# ax13.legend()
plt.tight_layout()
plt.savefig('theta_vs_epochs_in_9_6_pt6.pdf')
plt.show()



### V0, X0, SIG, LOSS, THETA 3D PLOT ###
 
# plotting using interpolation to hopefully avoid the spikes in the surface plot
x,y,z = v0_x0_sig_percentageloss_theta[:,2], v0_x0_sig_percentageloss_theta[:,0], v0_x0_sig_percentageloss_theta[:,4]
x_grid, y_grid = np.meshgrid(np.linspace(min(x), max(x), 25), np.linspace(min(y), max(y), 25)) # grid of data where i want to interpolate
z_grid = griddata((x,y),z, (x_grid,y_grid), method='linear') # interpolating the data

fig3 = plt.figure(figsize=(l,h))
ax3 = fig3.add_subplot(111, projection='3d')
surf3 = ax3.plot_surface(x_grid, y_grid, z_grid, cmap='viridis', linewidth=0, antialiased=False)
fig3.colorbar(surf3, ax=ax3)
# ax3.set_xlabel('Loss (%)')
ax3.set_xlabel('$\sigma$')
ax3.set_ylabel('$V_0$')
# ax3.set_ylabel('$x_0$')
# ax3.set_ylabel('$\sigma$')
ax3.set_zlabel(r'$\theta$', rotation=0) 
# ax3.set_zticklabels([])
plt.tight_layout()
plt.savefig('theta_surface_in_9_6_pt6.pdf')
plt.show()















# used to have these after optimiser.step() in the training loop

        # print('==================================================================')
        # outputs1 = model(inputs)
        # # print('outputs1 = ',outputs1)
        # funcderiv = torch.autograd.grad(outputs1.sum(),inputs, retain_graph=True)[0]  # this should be the functional derivative of Tml[rho]
        # # ^ this is going to have to go into the main block as we are redefining the loss entirely. we want the system to be trained in a different way
        # # print(funcderiv)
        # funcderiv_plus_v = funcderiv + pots # this is the functional derivative of Tml plus the potential 
        # const_vec = -torch.ones_like(funcderiv_plus_v) # this is the vector of -1's that is the same shape as the functional derivative
        # theta = torch.arccos((const_vec*funcderiv_plus_v).sum(dim=1)/(torch.norm(const_vec,dim=1)*torch.norm(funcderiv_plus_v,dim=1))) # this is the expression for theta which gives the error between the funcderiv_plus_v and the lagrange multiplier (which is the chemical potential i think) which is a constant in the form of an angle theta. this angle theta has as many entries as terms in each inputs i.e. the batch size
        # print(f'theta = {theta} \nlen(theta) = {len(theta)}')
        # print('==================================================================')
        
        # if i==0 : #and (epoch ==0 or epoch==50): 
        #     print('==================================================================')
        #     print(f'epoch={epoch}')
        #     print('input grad = ',inputs.grad)
        #     print('target grad = ',target.grad)
        #     outputs1 = model(inputs)
        #     outputs1_grad = torch.autograd.grad(outputs1.sum(),inputs, retain_graph=True)[0]  # this should be the functional derivative of Tml[rho]
        #     print(outputs1_grad) 
        #     print('len = ',len(outputs1_grad[0]))
        #     print('==================================================================')
        # i=i+1
