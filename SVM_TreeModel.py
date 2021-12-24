import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from sklearn import svm
import math
import pandas as pd

class node:
    def __init__(self,left=None,right=None,weight=None):
        self.left = left
        self.right = right
        self.weight = weight
        self.left_present = False
        self.right_present = False
    def predict(self, X):
        X_ = X
        if (self.left_present):
            Xa = self.left.predict(X)
            X_ = np.hstack((X, Xa.reshape((Xa.shape[0], 1))))
        if (self.right_present):
            Xb = self.right.predict(X)
            X_ = np.hstack((X_, Xb.reshape((Xb.shape[0], 1))))
        assert(self.feature_size == X_.shape[1])
        assert(X.shape[0] == X_.shape[0])

        return self.clf.predict(X_)
    def solveSVM(X_, Y):
        self.clf = svm.SVC(kernel='linear', C = 1e-15)
        self.clf.fit(X_, Y)
        self.feature_size = X_.shape[1]


def solveLPP(A,B,c):
    ans = linprog(c, A_eq=A, b_eq=B, method='simplex')
    return ans


def solLPP1(X,Y):
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

    w = solLPP1(X,Y)
    n = X.shape[0]

    i = 0
    nc3 = 0
    nc4 = 0
    Xa = np.zeros((n, 1), dtype=float)
    Xb = np.zeros((n, 1), dtype=float)
    pred = np.dot(X, w)
    while(i < n):
        Y_pred_01 = np.where(pred > 0, 1, 0)
        if(Y_pred_01[i] == 0 and Y[i] == 1):
            nc3 += 1
            Xa[i][0] = 1
        elif(Y_pred_01[i] == 1 and Y[i] == 0):
            nc4 += 1
            Xb[i][0] = 1

        i += 1

    r1 = node()
    # print(f"{nc3} samples in class c3,", ("not" if (nc3==0) else ""), "creating a new neuron for class c3")
    # print(f"{nc4} samples in class c4,", ("not" if (nc4==0) else ""), "creating a new neuron for class c4")
    X_ = X
    if (nc3 > 0):
        X_ = np.hstack((X, Xa))
        r1.left_present = True
    if (nc4 > 0):
        X_ = np.hstack((X_, Xb))
        r1.right_present = True
    r1.solveSVM()

    if(nc3 > 0):
        qw = int(np.sum(np.where(Y==Y_pred_01,Y,0)))
        YA = np.zeros(n-qw)
        X_new = np.zeros((n-qw,X.shape[1]))
        i = 0
        j = 0
        while(i < n):
            if(Y[i] == 0 and Y_pred_01[i] == 0):
                YA[j] = 0
                X_new[j] = X[i]
                j = j+1
            elif(Y[i] == 1 and Y_pred_01[i] == 0):
                YA[j] = 1
                X_new[j] = X[i]
                j = j+1
            elif(Y[i] == 0 and Y_pred_01[i] == 1):
                YA[j] = 0
                X_new[j] = X[i]
                j = j+1

            i = i+1
        r1.left = solve(X_new,YA)


    if (nc4 > 0):
        qw = int(np.sum(np.where(Y == Y_pred_01,Y,1)))
        YB = np.zeros(qw)
        X_new = np.zeros((qw,X.shape[1]))
        i = 0
        j = 0
        while(i < n):
            if(Y[i] == 1 and Y_pred_01[i] == 1):
                YB[j] = 0
                X_new[j] = X[i]
                j = j+1
            elif(Y[i] == 1 and Y_pred_01[i] == 0):
                YB[j] = 0
                X_new[j] = X[i]
                j = j+1
            elif(Y[i] == 0 and Y_pred_01[i] == 1):
                YB[j] = 1
                X_new[j] = X[i]
                j = j+1

            i = i + 1
        r1.right = solve(X_new, YB)
    return r1





# def transform_data(df):
#     df["y"] = df["diagnosis"].apply(lambda x: 1 if (x=="M") else 0)
#     df.drop(columns= ["diagnosis", "id"], inplace = True)
#     return df
def transform_data(df):
    df["y"] = df["class"].apply(lambda x: 1 if (x=="present") else 0)
    df.drop(columns = ["class"], inplace=True)
    return df
if __name__ == '__main__':
    dftrain = pd.read_csv('data/heart-statlog_csv.csv', header = 0)

    dftrain = transform_data(dftrain)

    Xtrain = dftrain.iloc[:,:-1].to_numpy()
    Xtrain = np.c_[(np.ones(Xtrain.shape[0]),Xtrain)]
    ytrain = dftrain.iloc[:,-1].to_numpy()
    n = Xtrain.shape[0]
    i = 0
    k = 10
    r = int(n/10)
    sum = 0
    while(i < k):
        Xtra1 = Xtrain[0 : r*i]
        Ytra1 = ytrain[0 : r*i]
        Xtra2 = Xtrain[r*(i+1) : ]
        Ytra2 = ytrain[r*(i+1) : ]
        Xtra = np.concatenate((Xtra1, Xtra2),axis=0)
        Ytra = np.concatenate((Ytra1, Ytra2),axis=0)
        Xtest = Xtrain[r*i : r*(i+1)]
        Ytest = ytrain[r*i : r*(i+1)]

        root = solve(Xtra,Ytra)
        prediction = root.predict(Xtest)
        acc = np.sum(np.where(Ytest == prediction, 1, 0))/ Ytest.shape[0]
        print(f"The accuracy for fold {i} is {acc}")
        sum = sum + acc
        i = i + 1
    print(f"The overall accuracy is {sum/k}")
    # ts = 230
    # Xtra = Xtrain[:ts]
    # Ytra = ytrain[:ts]
    # Xtest = Xtrain[ts:]
    # Ytest = ytrain[ts:]
    #
    # root = solve(Xtra,Ytra)
    # prediction = root.predict(Xtest)
    # acc = np.sum(np.where(Ytest == prediction,1,0))/Ytest.shape[0]
