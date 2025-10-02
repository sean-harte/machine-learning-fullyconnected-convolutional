#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 17:23:04 2025

@author: seanharte
"""

# I have no idea why but at N=1625 the results of the program strts to get very weird

import numpy as np
import matplotlib.pylab as plt
import scipy as sc
from scipy import integrate

# trying to create code that solves for the kinetic energy of a particle in a box

#constants
hbar = 1
mass = 1 #mass
a = 1 #length of box

# no. of electrons
n = 1

#number of points (number of gaps = N-1)
N = 2000

# discretisation of box
x_array = np.linspace(0,1,N)
dx = 1/(N-1)

# this is for the analytical energy levels of a particle in a box. no potential
En = np.zeros(N)
for i in range(N):
    En[i] = ((hbar**2)*((i+1)**2)*(np.pi**2))/((2*mass)*(a**2))


#first derivative matrix
ddxM = np.diag(-np.ones(N-1),-1) + np.diag(np.ones(N-1), 1)

#second derivative matrix
d2dxM = np.diag(-2*np.ones(N)) + np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1)

# kinetic energy operator matrix (applied to psi)
T = -((hbar**2)/(2*mass*(dx**2)))*d2dxM



# we now define initial and final values that we want to have as the range for our dataset
# we are then going to put the rest of the code in 3 for loops.
# one loop for each variable we want to get a range for.
# we are then going to record v0,x0,sig,rho[0],Trho[0] and Erho[0]. all the data for 
# the ground state of rho.
v0i = 140
v0f = 150
v0_n = 5
v0_array = np.linspace(v0i,v0f,v0_n)

x0i = 0.5
x0f = 0.5
x0_n = 1
x0_array = np.linspace(x0i,x0f,x0_n)

sigi = 0.05
sigf = 0.15
sig_n = 3
sig_array = np.linspace(sigi,sigf,sig_n)

xdim = v0_n*x0_n*sig_n

t_norm_array = np.zeros(xdim)
dataset = np.zeros((xdim,5 + N + N)) # scalar dataset. order: v0,x0,sig,Trho,Erho,rho,potential. dataset for NN with the functional derivative
dataset_no_pots = np.zeros((xdim,5 + N)) # dataset for NN without the func_deriv
rho_dataset = np.zeros((xdim,N)) # vector dataset of just rho
for m in range(sig_n):
    for k in range(x0_n):
        for j in range(v0_n):
            # define the gaussian potential matrix
            v0 = v0_array[j]
            x0 = x0_array[k]
            sig = sig_array[m]
            def v(x):
                v = -v0*np.exp(-((x-x0)**2)/(2*(sig**2)))
                return v
            
            va = np.zeros(N)
            for i in range(N):
                va[i] = v(x_array[i])
            V = np.diag(va)
            
            Vtest = np.diag(np.zeros(N))
            
            # define the Hamiltonian
            H = T + V
            
            # getting the eigenvalues of the matrix H to get the kinetic energy soon
            eig_val, eig_vec = np.linalg.eigh(H)
            
            # eigenvalues are not sorted in increasing order. Code below sorts them in increasing order
            # eigenvectors also sorted in order of increasing eigenvalue
            # inds is the order of indices used to sort the eigenvaues into increasing order
            inds = eig_val.argsort()
            
            eig_val = eig_val[inds]
            eig_vec = eig_vec[:,inds]
            
            # have to normalise the eigenvectors in the quantum mechanical sense not the linear algebra sense
            for i in range(N):
                eig_vec[:,i] = (1/np.sqrt(integrate.simpson(y=eig_vec[:,i]**2,x=x_array)))*eig_vec[:,i]
            
            # T - the Thomas-Fermi kinetic energy
            t_array = np.zeros(N)
            for i in range(N):
                t = (hbar**2/(2*mass*(2*dx)**2))*((ddxM@eig_vec[:,i])**2)# still a vector rn
                a = integrate.simpson(y=t,x=x_array)
                t_array[i] = t_array[i] + a
            t_norm = 0
            for i in range(n):
                t_norm = t_norm + t_array[i]
            
            # we now have code that finds and sorts the eigenvectors in order of increasing eigenvalue.
            # we need to get the kinetic energies for each eigenstate/wavefunction
            # to do this we get the expected value of the T matrix using the eigenvector as the wavefunction
            # this equation reduces to the hbar**2/2*m|nabla*phi|**2 (eigenvector). we already have 
            # a matrix for nabla which is ddxM. This function is the summed over every level that has an 
            # electron in it.
            # list1 can be used to tell the loop in what energy levels the electrons are in and
            # therefore if there are multiple electrons in a level the code will add all the energies 
            # together rather than replacing the energy of the n-1th electron with the nth electron
        
            
            # rho
            # we now make the array of electron densities which is the eigenvector squared for each
            # eigenvalue.
            # i am doing this by squaring every term in every array. can cut down on computation time 
            # if we just need the ground state by just squaring the ground state eigenvector
            rho = eig_vec**2
            # rho is now an array of electron densities 
            # these are normalised
            rho_0 = np.zeros(N)
            for i in range(n):
                rho_0 = rho_0 + rho[:,i]
            
            # T[rho] - the von Weiskracker kinetic energy
            # now need to get array for T[rho] - kinetic energy as a function of electron density.
            # this expression is an proportional to the integral of the derivative of rho wrt x squared 
            # divided by rho.
            # we have a matrix that gives us the first derivative == ddxM.
            Trho = np.zeros(N)
            for i in range(N):
                a = (1/(2*dx))*(ddxM@(rho_0))
                b = a**2
                c = b/rho_0
                d = integrate.simpson(y=c,x=x_array)
                f = (1/8)*d # this formula is definitely missing its factors of hbar and m
                
                Trho[i] = f

            
            # E[rho]
            # now we need to define the total energy, which is a functional of the electron density rho,
            # as the sum of the kinetic energy functional and the integral of the potential*rho
            Erho = np.zeros(N)
            for i in range(N):
                a = V@rho_0
                b = integrate.simpson(y=a,x=x_array)
                Erho[i] = Trho[i] + b
            
            
            index = j + v0_n*k + v0_n*x0_n*m
            dataset[index, 0] = v0
            dataset[index, 1] = x0
            dataset[index, 2] = sig
            dataset[index, 3] = Trho[0]
            dataset[index, 4] = Erho[0]
            # dataset[index, 5:] = rho[:,0]
            dataset[index,5:5+N] = rho_0
            dataset[index,5+N:5+N+N] = va
            # rho_dataset[index] = rho[:,0]
            rho_dataset[index] = rho_0
            
            dataset_no_pots[index, 0] = v0
            dataset_no_pots[index, 1] = x0
            dataset_no_pots[index, 2] = sig
            dataset_no_pots[index, 3] = Trho[0]
            dataset_no_pots[index, 4] = Erho[0]
            # dataset_no_pots[index, 5:] = rho[:,0]
            dataset_no_pots[index,5:] = rho_0
            
            t_norm_array[index] = t_norm # this is for the kinetic energy calculated using the expectation value of the T matrix using the eigenvectors.
            #print(index)
            # if (index+1)%100==0:
            #     print(f'Index = {index+1}/{xdim}')
            print(f'Index = {index+1}/{xdim}')
        
# =============================================================================
#             plt.plot(x_array, rho_dataset[index], label = '$rho_0$, index = ' + str(index))
#             plt.legend()
#             plt.show()
# =============================================================================

            # print('t_TF = ', t_TF, 't_array[0] = ', t_array[0], 't_array[1] = ', t_array[1], 't_array[2] = ', t_array[2])
            # print(index,'ratio = ', Trho[0]/t_array[0])
            
            #in report have a plot of the relation between von Weiskracker and Thomas-Fermi KE's

# np.save('dataset',dataset)
# np.save('dataset_no_pots',dataset_no_pots)

np.save('dataset_large',dataset)

# plt.plot(dataset[:,3]/t_norm_array)

# print('Trho/t_from_eig = ', )
