#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Wayne Isaac Tan Uy, PhD
28 Dec 2020
"""

import numpy as np
import numpy.linalg as LA

def genSPDGaussMatrix(N,diagVals,mu,stdev):
    """
    Generate symmetric positive definite matrix using realizations from a 
    Gaussian random variable.

    Parameters
    ----------
    N : size of the square matrix.
    diagVals : N x N square diagonal matrix whose entries are the target eigenvalues.
    mu : mean of the Gaussian random variable.
    stdev : standard deviation of the Gaussian random variable.

    Returns
    -------
    A : N x N symmetric positive definite matrix with eigenvalues contained in diagVals.

    """
    
    GaussMat = np.random.normal(mu,stdev,size = (N,N))
    GaussMat = 0.5*(GaussMat + GaussMat.T)
    _, eigVec = LA.eig(GaussMat)
    A = eigVec.T @ diagVals @ eigVec
    A = 0.5 * ( A + A.T )
    
    return A

def genSPDUnifMatrix(N,diagVals,upperbnd):
    """
    Generate symmetric positive definite matrix using realizations from a 
    uniform random variable.

    Parameters
    ----------
    N : size of the square matrix.
    diagVals : N x N square diagonal matrix whose entries are the target eigenvalues.
    upperbnd : upper bound for the range of the uniform random variable.

    Returns
    -------
    A : N x N symmetric positive definite matrix with eigenvalues contained in diagVals.

    """
    
    UnifMat = np.random.uniform(0,upperbnd,size = (N,N))
    UnifMat = 0.5*(UnifMat + UnifMat.T)
    _, eigVec = LA.eig(UnifMat)
    A = eigVec.T @ diagVals @ eigVec
    A = 0.5 * ( A + A.T )
    
    return A    
    
def genPartialObsMat(po,N):
    """
    Generate the selction matrix and its complement.

    Parameters
    ----------
    po : percentage of observed state components (from 0 to 100).
    N : dimension of the fully observed system.

    Returns
    -------
    poID : indices of the full state that are observed.
    poIDC : indices of the full state that are not observed.
    Po : selection matrix that selects the observed states from the full state.
    Poperp : complement of the selection matrix.

    """
    
    poID = np.ceil(np.linspace(0, N-1, num = np.floor(N*po/100).astype(int))).astype(int)
    poIDC = np.setdiff1d(np.arange(N), poID)

    # Selection matrix
    Po = np.eye(N)[poID,:] 
    Poperp = np.eye(N)[poIDC,:]

    return poID, poIDC, Po, Poperp

def TimeStepMarkovianSys(Sys,signal,xInit):
    """
    Time-step system Markovian system x(k+1) = Ax(k) + Bu(k).
    
    Parameters
    ----------
    Sys : dictionary representing a LTI system.
    signal : 2-d array for control input.
    xInit : array for initial condition.
    
    Returns
    -------
    XTraj : 2-d array representing state trajectory.
    
    """
    
    if 'A' in Sys:
        A = Sys['A']
        B = Sys['B']
    elif 'Ar' in Sys:
        A = Sys['Ar']
        B = Sys['Br']

    nSteps = signal.shape[1]
    XTraj = np.zeros((xInit.shape[0],nSteps + 1))
    XTraj[:,0:1] = xInit
    
    for k in range(nSteps):
        XTraj[:,k+1:k+2] = A @ XTraj[:,k:k+1] + B @ signal[:,k:k+1]
    
    return XTraj

def genBasis(Sys,signal,xInit,rdim,Po):
    """
    Generate basis matrix via POD from the observed states.
    
    Parameters
    ----------
    Sys : dictionary representing a LTI system.
    signal : 2-d array for control input.
    xInit : array for initial condition.
    rdim : reduced dimension.
    Po: selection matrix that selects the observed states from the full state.
    
    Returns
    -------
    V : basis matrix whose columns are basis vectors.
    
    """
    
    nIC = xInit.shape[1]
    
    XTrajAll = np.array([])
    
    for k in range(nIC):
        
        XTraj = TimeStepMarkovianSys(Sys,signal,xInit[:,k:k+1])
        XTraj = XTraj[Po,:]
        
        if XTrajAll.shape[0] == 0:
            XTrajAll = XTraj
        else:
            XTrajAll = np.hstack((XTrajAll, XTraj))
            
    # compute svd
    _ , _ , Vh = LA.svd(XTrajAll.T)
    Vtmp = Vh.T
    V = Vtmp[:, :rdim]
    Vperp = Vtmp[:,rdim:]
    
    return V, Vperp

def genMarkovianSys(Sys,V):
    """
    Generate Markovian reduced model operators 'Ar', 'Br' in xr(k+1) = Ar xr(k) + Br u(k)
    via projection.
    
    Parameters
    ----------
    Sys : dictionary representing a LTI system.
    V : basis matrix whose columns are basis vectors.
    
    Returns
    -------
    redModel : dictionary with keys 'Ar', 'Br'.
    
    """
    
    A = Sys['A']
    B = Sys['B']
    
    redModel = dict()
    redModel['Ar'] = V.T @ (A @ V)
    redModel['Br'] = V.T @ B
    
    return redModel

def genNonMarkovianModel(Sys,ROM,Q,Qperp,lag):
    """
    Generate operators of the reduced model with non-Markovian term

    Parameters
    ----------
    Sys : dictionary representing the full LTI system.
    ROM : dictionary representing the Markovian reduced system.
    Q : projection from full state to the reduced observed state.
    Qperp : orthogonal complement of Q.
    lag : lag of the non-Markovian term.

    Returns
    -------
    nonMarkovianROM : a dictionary with keys 'StateOp' and 'SignalOp' which are lists
                      representing the Markovian and non-Markovian operators 
                      that are coefficients of the state and input.

    """
    
    A = Sys['A']
    B = Sys['B']
    Ar = ROM['Ar']
    Br = ROM['Br']
    
    StateOp = []
    SignalOp = []
    
    StateOp.append(Ar)
    SignalOp.append(Br)
    
    QTAQperp = Q.T @ A @ Qperp
    QperpTAQ = Qperp.T @ A @ Q
    QperpTB = Qperp.T @ B
    QperpTAQperp = Qperp.T @ A @ Qperp

    tempMat = np.eye(Qperp.shape[1])
    
    for k in range(lag):
        StateOp.append( QTAQperp @ tempMat @ QperpTAQ )
        SignalOp.append( QTAQperp @ tempMat @ QperpTB )
        
        tempMat = tempMat @ QperpTAQperp
    
    nonMarkovianROM = dict()
    nonMarkovianROM['StateOp'] = StateOp
    nonMarkovianROM['SignalOp'] = SignalOp
    
    return nonMarkovianROM

def TimeStepNonMarkovianSys(nonMarkovianROM,signal,xInit):
    """
    Time step the non-Markovian system.

    Parameters
    ----------
    nonMarkovianROM : a dictionary of lists containing the Markovian and non-Markovian operators.
    signal : 2-d array for control input.
    xInit : array for initial condition.

    Returns
    -------
    XTraj : 2-d array representing state trajectory.

    """
    
    StateOp = nonMarkovianROM['StateOp']
    SignalOp = nonMarkovianROM['SignalOp']
    
    nOper = len(StateOp)
    
    # convert StateOp and SignalOp to arrays
    
    StateOp = np.asarray(StateOp)
    SignalOp = np.asarray(SignalOp)
    
    nSteps = signal.shape[1]
    XTraj = np.zeros((xInit.shape[0],nSteps + 1))
    XTraj[:,0:1] = xInit
    
    for k in range(nSteps):
        
        StateOpTrunc = StateOp[:k+1,:,:]
        SignalOpTrunc = SignalOp[:k+1,:,:]
        
        # if time step is smaller than lag
        if k + 1 <= nOper: 
            StateTrunc = XTraj[:,k::-1]
            SignalTrunc = signal[:,k::-1]
        
        # if time step exceeds lag
        else:
            StateTrunc = XTraj[:,k:k-nOper:-1]
            SignalTrunc = signal[:,k:k-nOper:-1]
        
        XTrajTemp = np.einsum("ijk,ki->ji",StateOpTrunc,StateTrunc) + \
                    np.einsum("ijk,ki->ji",SignalOpTrunc,SignalTrunc)
                    
        XTraj[:,k+1:k+2] = np.sum(XTrajTemp, axis = 1).reshape(-1,1)
        
    return XTraj