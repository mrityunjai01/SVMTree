import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from sklearn import svm
import math
import pandas as pd
from utils.measure_time import timing
LP_METHOD = "highs-ds"
EPSILON = 1e-12
verbose = True
class node:
    def __init__(self,left=None,right=None,weight=None):
        self.left = left
        self.right = right

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
        try:
            product = np.matmul(X_, self.w) + self.b

        except Exception as e:
            print(e)

            raise Exception("prediction error, cant predict.")
        preds = np.where(product<0, -1, 1)
        return preds
    def solveMaximumMargin(self, X_, Y):

        n, d = X_.shape
        self.feature_size = d
        c = np.ones(1 + 2 * d)
        c[0] = 0

        b = -np.ones (n)
        A = np.zeros((n, 1 + 2 * d))
        Y = np.expand_dims(Y, axis = 1)
        A[:, :1] = -1
        A[:, 1:(d+1)] = -X_
        A[:, d+1:] = X_
        A = A * Y
        bounds = [(0, None) for i in range (1 + 2 * d)]
        bounds[0] = (None, None)

        self.ans = linprog(c, A_ub=A, b_ub=b, method=LP_METHOD, bounds = bounds)


        if (self.ans.status > 1):
            print("cant solve svm, status = ", self.ans.status)
            print(X_.shape, Y.shape)

            raise Exception("Cant solve the svm")
        else:
            self.w = self.ans.x[1:(d+1)] + self.ans.x[(d+1):]
            self.b = self.ans.x[0]
    def getHyperPlaneFromTwoPoints(self, x1, x2):
        assert(x1.shape[0]==x2.shape[0])
        d = x1.shape[0]
        self.w = (2 + 1e-8) * (x2 - x1) / (np.linalg.norm(x1 - x2) ** 2)
        self.b = - np.dot(self.w , (0.5 * (x1 + x2)))
        assert(np.dot(x1, self.w) + self.b <= -1)
        assert(np.dot(x2, self.w) + self.b >= 1)
        # print("assertions satisfied")
def iterate(A, B, C, n, d):

    bounds = [(0, None) for i in range (d + 1 + n)]
    for i in range (d+1):
        bounds[i] = (None, None)
    ans = linprog(C, A_ub=A, b_ub=B, method=LP_METHOD, bounds = bounds)
    if (ans.status > 1):
        print("cant solve lp, code", ans.status)
        print(X.shape, Y.shape)
        raise Exception("Cant solve the first lp")
        return
    return ans.x

def solveLP1(X,Y):
    n, d = X.shape

    A = np.zeros((n, d + 1 + n))
    B = -np.ones(n)

    C = np.zeros(d + 1 + n)
    C[-n:] = np.ones(n)

    Y = np.expand_dims(Y, axis = 1)

    A[:, -n:] = -np.eye(n)
    A[:, 1:-n] = -(X * Y)
    A[:, :1] = - Y
    max_iterations = 15
    i = 0
    def coeff_map(x, epsilon = EPSILON):
        # return x
        return 1/(epsilon + x)
        np.vectorize(coeff_map)
    while (i < max_iterations):
        try:
            x = iterate(A, B, C, n, d)
        except Exception as e:
            print(e)
            print(f"cant solve LP at iteration {i}")
            break

        q = x[-n:]
        # print(f"The sum of q_i's at iteration {i} is {np.sum(q)}, number of nonzero q_i's is {np.count_nonzero(q)}")
        if (np.sum(q) > 0):
            C[-n:] = np.array([coeff_map(x) for x in q])
        else:
            # print(f"solved the LP perfectly, i.e. no outliers")
            return x[1:(d+1)], x[:1]
            break

        i += 1
        if (i==max_iterations):
            # print("max iterations reached, still not perfect, some outliers")
            return x[1:(d+1)], x[:1]



@timing
def solve(X, Y, w=None, b=None):
    print("New  neuron made")
    if (verbose):
        print(f"Neuron received, {X.shape[0]} samples")
    if (w is None):
        try:
            w, b = solveLP1(X,Y)

        except Exception as e:
            print(e)
            print("Error in first lp function, solveLP1")
            quit()
    else:
        print("Neuron made to do 2 point hyperplane thing")

    n = X.shape[0]

    i = 0
    nc1 = nc2 = nc3 = nc4 = 0
    last_c1_sample = last_c2_sample = last_c3_sample = last_c4_sample = 0
    c3_mask = [False for i in range (n)]
    c4_mask = [False for i in range (n)]
    Xa = np.zeros((n, 1), dtype=float)
    Xb = np.zeros((n, 1), dtype=float)
    pred = np.dot(X, w) + b
    Y_pred_01 = np.where(pred >= 0, 1, -1)
    while(i < n):
        if(Y_pred_01[i] == -1 and Y[i] == 1):
            nc3 += 1
            Xa[i][0] = 1
            c4_mask[i] = True
            last_c3_sample = i
        elif(Y_pred_01[i] == 1 and Y[i] == -1):
            nc4 += 1
            Xb[i][0] = 1
            c3_mask[i] = True
            last_c4_sample = i
        elif(Y_pred_01[i] == -1 and Y[i] == -1):
            nc1 += 1
            c4_mask[i] = True
            c3_mask[i] = True
            last_c1_sample = i
        else:
            nc2 += 1
            c4_mask[i] = True
            c3_mask[i] = True
            last_c2_sample = i

        i += 1
    if (verbose):
        print(f"class sizes, c1 = {nc1}, c2 = {nc2}, c3 = {nc3}, c4 = {nc4}\n")
    r = node()

    X_ = X
    Y_ = Y
    if (nc2 == 0 and nc1 == 0):
        if (verbose):
            print("both c1 and c2 are empty")
        raise Exception ("both c1 and c2 are empty")

    if (nc3 > 0 and nc2 > 0):
        X_ = np.hstack((X_, Xa))
        r.left_present = True
    elif (nc3 > 0):
        print("c2 is empty, getting perpendicular bisecting hyperplane")

        x1 = X[last_c1_sample]
        x2 = X[last_c3_sample]

        r.getHyperPlaneFromTwoPoints(x1, x2)
        return solve(X, Y, r.w, r.b)

    if (nc4 > 0 and nc1 > 0):
        X_ = np.hstack((X_, Xb))
        r.right_present = True
    elif (nc4 > 0):
        x1 = X[last_c4_sample]
        x2 = X[last_c2_sample]
        print("c1 is empty, getting perpendicular bisecting hyperplane")

        r.getHyperPlaneFromTwoPoints(x1, x2)
        return solve(X, Y, r.w, r.b)



    try:
        if (verbose):
            print(f"putting in, {X_.shape[0]} samples for svm")
        r.solveMaximumMargin(X_, Y_)
    except Exception as e:
        print(e)
        print("There's an error, terminating.")
        quit()

    if(nc3 > 0 and nc2 > 0):
        n = X.shape[0]
        size_of_left_neuron = nc1 + nc3 + nc4
        if (verbose):
            print(f"creating left neuron with {size_of_left_neuron} out of {n} samples")
        YA = np.zeros(size_of_left_neuron)
        X_new = np.zeros((size_of_left_neuron, X.shape[1]))
        i = 0
        j = 0
        while(i < n):
            if(Y[i] == -1 and Y_pred_01[i] == -1):
                YA[j] = -1
                X_new[j] = X[i]
                j = j+1
            elif(Y[i] == 1 and Y_pred_01[i] == -1):
                YA[j] = 1
                X_new[j] = X[i]
                j = j+1
            elif(Y[i] == -1 and Y_pred_01[i] == 1):
                YA[j] = -1
                X_new[j] = X[i]
                j = j+1

            i += 1
        r.left = solve(X_new, YA)

    else:
        if (verbose):
            print("cant create left neuron")
    if (nc4 > 0 and nc1 > 0):
        n = X.shape[0]
        size_of_right_neuron = nc2 + nc3 + nc4
        qw = int(np.sum(np.where(Y==0, 1, 0) * np.where(Y_pred_01==0, 1, 0)))
        if (verbose):
            print(f"creating right neuron with {size_of_right_neuron} out of {n} samples")
        YB = np.zeros(size_of_right_neuron)
        X_new = np.zeros((size_of_right_neuron, X.shape[1]))
        i = 0
        j = 0
        while(i < n):
            if(Y[i] == 1 and Y_pred_01[i] == 1):
                YB[j] = -1
                X_new[j] = X[i]
                j = j+1
            elif(Y[i] == 1 and Y_pred_01[i] == -1):
                YB[j] = -1
                X_new[j] = X[i]
                j = j+1
            elif(Y[i] == -1 and Y_pred_01[i] == 1):
                YB[j] = 1
                X_new[j] = X[i]
                j = j+1

            i += 1
        r.right = solve(X_new, YB)
    else:
        if (verbose):
            print("cant create right neuron")
    return r




def transform_data(df):
    df["y"] = df["class"].apply(lambda x: 1 if (x=="present") else -1)
    df.drop(columns = ["class"], inplace=True)
    return df
if __name__ == '__main__':
    dftrain = pd.read_csv('data/heart-statlog_csv.csv', header = 0)

    dftrain = transform_data(dftrain)
    print(f"data size = {dftrain.shape}")
    Xtrain = dftrain.iloc[:,:-1].to_numpy()

    ytrain = dftrain.iloc[:,-1].to_numpy()
    n = Xtrain.shape[0]
    i = 0
    k = 1
    r = int(n/10)
    sum = 0
    while(i < k):
        Xtra1 = Xtrain[0 : r*i]
        Ytra1 = ytrain[0 : r*i]
        Xtra2 = Xtrain[r*(i+1) : ]
        Ytra2 = ytrain[r*(i+1) : ]
        Xtra = np.concatenate((Xtra1, Xtra2), axis=0)
        Ytra = np.concatenate((Ytra1, Ytra2), axis=0)
        Xtest = Xtrain[r*i : r*(i+1)]
        Ytest = ytrain[r*i : r*(i+1)]

        root = solve(Xtra, Ytra)
        try:
            prediction = root.predict(Xtest)
        except Exception as e:
            print(e)
            print(f"cant get prediction for fold {i}, moving ahead")
            continue
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
