#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wayne Isaac Tan Uy, PhD
28 Dec 2020
"""
import numpy as np
import numpy.linalg as LA
from utils import *
import matplotlib.pyplot as plt

Data = np.load('LowPartialObsLowDim.npz')

np.random.seed(1)

# simulation parameters

N = 10 # dimension of full system
nLag = 1 # lag of non-Markovian model
nSteps = 50 # simulation time steps
rdim = 2 # dimension of reduced observed state
po = 30 # percentage of observed states

# generate selection matrix

poID, poIDC, Po, Poperp = genPartialObsMat(po,N)

# generate discrete full system

A = Data['A']
B = np.zeros((N,1))
Sys = {"A":A, "B":B}
signal = np.zeros((1, nSteps))

# generate reduced basis

V = Data['V']
Vperp = Data['Vperp']

# compute Markovian system

Q = Po.T @ V
Qperp = np.hstack((Po.T @ Vperp, Poperp.T))
MarkovianSys = genMarkovianSys(Sys,Q)

# generate initial condition

xInitTestQSpace = Data['xInitTestQSpace']
xInitTest = Q @ xInitTestQSpace
nInitTest = xInitTest.shape[1]

# compute Markovian system
MarkovianSysLag0 = genNonMarkovianModel(Sys,MarkovianSys,Q,Qperp,0)

# compute non-Markovian system with lag 1
nonMarkovianSysTrunc = genNonMarkovianModel(Sys,MarkovianSys,Q,Qperp,nLag)
    
# compute exact non-Markovian system
nonMarkovianSysExact = genNonMarkovianModel(Sys,MarkovianSys,Q,Qperp,nSteps)

# time step various models

ObservedTraj = []
MarkovianTraj = []
nonMarkovianTruncTraj = []
nonMarkovianExactTraj = []
MarkovianErr = np.zeros((nInitTest, nSteps + 1))   
nonMarkovianTruncErr = np.zeros((nInitTest, nSteps + 1))   
nonMarkovianExactErr = np.zeros((nInitTest, nSteps + 1))   
ProjErr = np.zeros((nInitTest, nSteps + 1))   

for k in range(nInitTest):
    
    # obtain observed trajectory
    XTraj = TimeStepMarkovianSys(Sys,signal, xInitTest[:,k:k+1])
    ZTraj = XTraj[poID,:]
    ObservedTraj.append(ZTraj)
    
    # obtain Markovian model trajectory
    ZrTraj = TimeStepNonMarkovianSys(MarkovianSysLag0,signal, Q.T @ xInitTest[:,k:k+1])
    MarkovianTraj.append(ZrTraj)
    
    # obtain non-Markovian model with lag 1 trajectory
    ZrTruncTraj = TimeStepNonMarkovianSys(nonMarkovianSysTrunc,signal, Q.T @ xInitTest[:,k:k+1])
    nonMarkovianTruncTraj.append(ZrTruncTraj)
    
    # obtain exact non-Markovian model trajectory
    ZrExactTraj = TimeStepNonMarkovianSys(nonMarkovianSysExact,signal, Q.T @ xInitTest[:,k:k+1])
    nonMarkovianExactTraj.append(ZrExactTraj)
    
    # compute error
    MarkovianErr[k:k+1,:] = LA.norm(ZTraj - V @ ZrTraj, axis = 0).reshape(1,-1)
    nonMarkovianTruncErr[k:k+1,:] = LA.norm(ZTraj - V @ ZrTruncTraj, axis = 0).reshape(1,-1)
    nonMarkovianExactErr[k:k+1,:] = LA.norm(ZTraj - V @ ZrExactTraj, axis = 0).reshape(1,-1)
    ProjErr[k:k+1,:] = LA.norm(ZTraj - V @ V.T @ ZTraj, axis = 0).reshape(1,-1)
    
#%%
    
# plot
IC = 1 # index for initial condition

fig, ax = plt.subplots()
ax.plot(np.arange(nSteps), MarkovianErr[IC,1:], label = 'Markovian model')
ax.plot(np.arange(nSteps), nonMarkovianTruncErr[IC,1:], label = 'non-Markovian model with lag 1')

ax.set_yscale("log")
ax.legend()
ax.set(xlabel='time', ylabel='reduced model error')
ax.grid()

plt.show()
    
fig, ax = plt.subplots()
ax.plot(np.arange(nSteps), nonMarkovianExactErr[IC,1:], label = 'exact non-Markovian model')
ax.plot(np.arange(nSteps), ProjErr[IC,1:], label = 'Projection error')

ax.set_yscale("log")
ax.legend()
ax.set(xlabel='time', ylabel='reduced model error')
ax.grid()

plt.show()
fig, ax = plt.subplots()
aveRelErr = (MarkovianErr[IC,1:] - nonMarkovianTruncErr[IC,1:])/LA.norm(ObservedTraj[IC][:,1:], axis = 0)
ax.plot(np.arange(nSteps), aveRelErr, label = 'err diff')
ax.set(xlabel='time', ylabel='difference in relative error')

ax.grid()
plt.show()
