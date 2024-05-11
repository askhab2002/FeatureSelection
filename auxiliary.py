'''
Functions for simulation artificial data.

'''



import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from pyitlib import discrete_random_variable as drv
from scipy.stats import chi2
import matplotlib.pyplot as plt

from FeatureSelection import Choice_Step, Choice

def f1(X_M: np.array, X_l: np.array, n: int):

    sum = 0.0
    for i in range(0, n):
        sum = sum + X_M[i] * X_l[i]

    return sum

def f2(X_M: np.array, X_l: np.array, n: int):

    sum = 0.0
    for i in range(0, n):
        sum += np.max((X_M[i], X_l[i]))

    return sum

def f3(X_M: np.array, X_l: np.array, n: int):

    sum = 0.0
    for i in range(0, n):
        sum += np.min((X_M[i], X_l[i]))

    return sum

def f4(X_M: np.array, X_l: np.array, n: int):

    sum = 0.0
    for i in range(0, n):
        if X_M[i] * X_l[i] < 0:
            sum += 1

    return sum

def f5(X_M: np.array, X_l: np.array, n: int):

    sum = 0.0
    for i in range(0, n):
        if X_M[i] * X_l[i] >  10**(6):
            sum += 1
        if X_M[i] * X_l[i] < -10**(-6):
            sum -= 1

    return sum

def f6(X_M: np.array, X_l: np.array, n: int):

    sum = 0.0
    for i in range(0, n):
        if X_M[i] > X_l[i]:
            sum += 1

    return sum


def sigma(X_M: np.array, F: float):
    sum_X = np.sum(X_M)

    return (1/(1 + np.exp(-(sum_X + F))))

def P_Y_1(X: np.array, effect: int, f_type: int):
    X_M = X[0:effect]
    X_l = X[effect:(2 * effect)]

    if f_type == 1:
        F = f1(X_M, X_l, effect)
    if f_type == 2:
        F = f2(X_M, X_l, effect)
    if f_type == 3:
        F = f3(X_M, X_l, effect)
    if f_type == 4:
        F = f4(X_M, X_l, effect)
    if f_type == 5:
        F = f5(X_M, X_l, effect)
    if f_type == 6:
        F = f6(X_M, X_l, effect)

    sig = sigma(X_M, F)

    return sig


def Generate_Y(X: np.array, n: int, f_type: int):
    P_1 = P_Y_1(X, n, f_type)
    P_0 = 1 - P_1

    return np.random.choice(np.array([0, 1]), p=[P_0, P_1])



def test(N_: int, p: int, M_: int, number_bins: int, f_type: int):
    
    mu, dispertion = 0, 1 # mean and standard deviation
    X_ = np.zeros((N_, p))
    Y = np.zeros(N_)



    for i in range(0, N_):
        X_[i] = np.random.normal(mu, dispertion, p)
        Y[i] = Generate_Y(X_[i], M_, f_type)


    X_b = np.zeros((N_, p))

#    number_bins = 2

    bins = np.quantile(X_, np.arange(0, 1, (1/number_bins)))
    X_b = np.digitize(X_, bins)

    for i in range(0, p):
        bins = np.quantile(X_[:, i], np.arange(0, 1, (1/number_bins)))
        X_b[:, i] = np.digitize(X_[:, i], bins)


    X_b = (np.rint(X_b)).astype(int)
    Y = (np.rint(Y)).astype(int)

    X_dim = number_bins
    Y_dim = 2

    dimensions = p * X_dim * (X_dim - 1) * (Y_dim - 1) + (1 - p) * (X_dim - 1) * (Y_dim - 1)

    quantile = chi2.ppf(0.95, dimensions)
    
    return Choice(X_b, Y, quantile)



def Plot_Features(p: int, M_: int, number_bins: int, min_N_: int, max_N_: int, number_N_: int, True_Features: np.array, f_type: int):
    N = np.linspace(min_N_, max_N_, number_N_, dtype=int)
    Succes_Rate = np.zeros(number_N_)
    False_Rate = np.zeros(number_N_)
 
    i = 0

    for N_ in N:
        Selected_Features = test(N_, p, M_, number_bins, f_type)

        Inter = np.intersect1d(Selected_Features, True_Features)

        Succes_Rate[i] = np.shape(Inter)[0] / np.shape(True_Features)[0]
        False_Rate[i] = (np.shape(Selected_Features)[0] - np.shape(Inter)[0]) / np.shape(Selected_Features)[0]

        i += 1


    fig, axs = plt.subplots(nrows= 1 , ncols= 2, figsize=(15, 7) )
    fig. suptitle('Feature selection with bins number = ' + str(number_bins) +  
              ', number of all features = ' + str(p) + ', number of relevant features = ' + str(2 * M_))

    axs[0].set_title('Positive Selection Rate')
    axs[0].set_xlabel('Sample size')
    axs[0].set_ylabel('PSR')
    axs[0].set_ylim(-0.1, 1.1)

    axs[1].set_title('False Selection Rate')
    axs[1].set_xlabel('Sample size')
    axs[1].set_ylabel('FSR')
    axs[1].set_ylim(-0.1, 1.1)

    axs[0].plot(N, Succes_Rate, marker='x', label='f_type = ' + str(f_type))
    axs[1].plot(N, False_Rate, marker='x', label='f_type = ' + str(f_type))

    plt.show()


