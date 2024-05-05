import numpy as np
from sklearn.metrics import mutual_info_score
from sklearn.feature_selection import mutual_info_classif
from pyitlib import discrete_random_variable as drv
from scipy.stats import chi2
import matplotlib.pyplot as plt






def Choice_Step(feature_indices: np.array,  X_b: np.array, Y: np.array, quantile: float):
    '''
    Parameters:

    feature_indices - set of indices of already choosen revelant features
    X_b             - array of feature samples
    Y               - targer array
    quantile        - distribution quantile

    Return:

    argmax          - index of new relevant feature
    True/False      - if True, then algorithm stops. If False, then algorithm continues
    
    '''
    p = np.shape(X_b)[1]
    n = np.shape(X_b)[0]
    relevant_number = np.shape(feature_indices)[0]

    all_indices = np.arange(0, p, dtype=int)

    
    gen = np.setdiff1d(all_indices, feature_indices)

    max_JMI = 0.0
    current_JMI = 0.0
    argmax = 0

    threshold = quantile #/(p - relevant_number)
    #print(threshold)
    
    for i in gen:
        for j in feature_indices:
            j = int(j)
            
            current_JMI += drv.information_mutual_conditional(Y, X_b[:, i], X_b[:, j])
        
        current_JMI += (1 - relevant_number) * mutual_info_score(X_b[:,i], Y)

        if current_JMI > max_JMI:
            max_JMI = current_JMI
            argmax = i
        
        current_JMI = 0
        
    if 2 * n * max_JMI < threshold:
        return argmax, True
    

    return argmax, False

def Choice(X_b: np.array, Y: np.array, quantile: float):
    '''
    Parameters:

    X_b             - array of feature samples
    Y               - targer array
    quantile        - distribution quantile

    Return:

    feature_indices - set of indices of relevant features
    
    
    '''

    feature_indices = []
    feature_indices = np.array(feature_indices)
    feature_indices = (np.rint(feature_indices)).astype(int)

    p = np.shape(X_b)[1]
    n = np.shape(X_b)[0]
    all_indices = np.arange(0, p, dtype=int)

    max_MI = 0
    current_MI = 0
    argmax = 0

    for i in all_indices:
        current_MI = mutual_info_score(X_b[:,i], Y)

        if current_MI > max_MI:
            max_MI = current_MI
            argmax = i

    feature_indices = np.insert(feature_indices, 0, argmax)

    all_indices = np.setdiff1d(all_indices, feature_indices)

    for i in all_indices:
        argmax, cond = Choice_Step(feature_indices, X_b, Y, quantile)

        if cond is True:
            break
        feature_indices = np.insert(feature_indices, 0, argmax)
    
    

    return feature_indices








