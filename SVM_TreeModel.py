import pulp as p
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import math
from cvxopt import matrix,solvers
from cvxopt.modeling import op
from cvxopt.modeling import variable
import pandas as pd

class node:
    def __init__(self,left=None,right=None,weight=None):
        self.left = left
        self.right = right
        self.weight = weight

def solveLPP(A,B,c):
    ans = linprog(c,A_eq=A,b_eq=B,method='simplex')
    return ans

def solLPP(X,Y):
    k = np.sum(np.where(Y == 0,1,0))
    m = np.sum(np.where(Y == 1,1,0))
    n,numf = X.shape

    A = np.zeros((n,2*numf+2*n))
    B = np.zeros(n)
    c = np.zeros(2*numf+2*n)

    i = 0
    v = 2*numf
    delta = 0.1

    while(i < n):
        j = 0
        while(j < numf):
            A[i][j] = X[i][j]
            j = j+1
        while(j < 2*numf):
            A[i][j] = -X[i][j-numf]
            j = j+1
        if(Y[i] == 0):
            A[i][v] = 1
            v = v+1
            A[i][v] = -1
            c[v] = 1/k
            v = v+1
            B[i] = -delta
        else:
            A[i][v] = -1
            v = v+1
            A[i][v] = 1
            c[v] = 1/m
            v = v+1
            B[i] = delta
        i = i+1

    ans = solveLPP(A,B,c)
    mat = ans.x
    w = np.zeros(numf)
    i = 0
    while(i < numf):
        w[i] = mat[i]-mat[i+numf]
        i = i+1
    return w

def solve(X,Y):
    w = solLPP(X,Y)
    n = X.shape[0]
    Yans = np.zeros(n,dtype=int)
    i = 0
    nc3 = 0
    nc4 = 0

    while(i < n):
        X1 = X[i]
        ans = np.dot(w,X1)
        if(ans > 0):
            Yans[i] = 1
        if(Yans[i] == 0 and Y[i] == 1):
            nc3 = nc3+1
        if(Yans[i] == 1 and Y[i] == 0):
            nc4 = nc4+1
        i = i+1

    r1 = node()
    print(nc3,nc4)

    if(nc3 > 0):
        qw = np.sum(np.where(Y==Yans,Y,0))
        YA = np.zeros(n-qw)
        X_new = np.zeros((n-qw,X.shape[1]))
        i = 0
        j = 0
        while(i < n):
            if(Y[i] == 0 and Yans[i] == 0):
                YA[j] = 0
                X_new[j] = X[i]
                j = j+1
            elif(Y[i] == 1 and Yans[i] == 0):
                YA[j] = 1
                X_new[j] = X[i]
                j = j+1
            elif(Y[i] == 0 and Yans[i] == 1):
                YA[j] = 0
                X_new[j] = X[i]
                j = j+1
            i = i+1
        r1.left = solve(X_new,YA)


    if(nc4 > 0):
        qw = np.sum(np.where(Y == Yans,Y,1))
        YB = np.zeros(qw)
        X_new = np.zeros((qw,X.shape[1]))
        i = 0
        j = 0
        while(i < n):
            if(Y[i] == 1 and Yans[i] == 1):
                YB[j] = 0
                X_new[j] = X[i]
                j = j+1
            elif(Y[i] == 1 and Yans[i] == 0):
                YB[j] = 0
                X_new[j] = X[i]
                j = j+1
            elif(Y[i] == 0 and Yans[i] == 1):
                YB[j] = 1
                X_new[j] = X[i]
                j = j+1
            i = i+1
        r1.right = solve(X_new,YB)
    
    return r1


dftrain = pd.read_csv('heart-statlog_csv.csv')
Xtrain = dftrain.iloc[:,:-1].to_numpy()
Xtrain = np.c_[(np.ones(Xtrain.shape[0]),Xtrain)]
ytrain = dftrain.iloc[:,-1].to_numpy()
ytrain = np.where(ytrain=='present',1,0)
Xtra = Xtrain[:230]
Ytra = ytrain[:230]
Xtest = Xtrain[230:]
Ytest = ytrain[230:]

root = solve(Xtra,Ytra)
