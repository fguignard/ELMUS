#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:24:23 2018

@author: fguignar1
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def _Activation_function(M, kind = 'logistic') :
    '''
    Activation functions for artificial neural network.
    
    Input : - M = np.matrix object
            - kind = kind of function (sigmoid, tanh, identity, relu, ...)
    
    Output : - M = np.matrix of transformed input matrix
    '''
    
    if kind == 'logistic' :
        M = 1/(1+ np.exp(-M))
        
    elif kind == 'relu' :
        M = np.array(M)
        M = M * (M > 0)
        M = np.matrix(M)
        
    elif kind == 'tanh' :
        M = np.tanh(M)
        
    elif kind == 'identity' :
        None
        
    else :
        print ('Erreur')
        print(fg)
    return M

class ELMRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, n_neurons=100, activation='logistic', random_state=None):
        self.n_neurons = n_neurons
        self.activation = activation
        self.random_state = random_state        

    def fit(self, X, y):
        '''
        Training for ELM
        Initialize random hidden weights and compute output weights
        
        Input : - X = #observations x #features, array
                - y = #observations, array
        
        Output : - self
        '''      
        
        X, y = check_X_y(X, y)
        
        self.X_ = X
        self.y_ = y
        
        X = np.matrix(X)
        y = np.matrix(y).transpose()
        n_obs, n_feat = X.shape
        
        # random drawing, seed
        np.random.seed(self.random_state)
        Drawing = np.random.uniform(-1, 1, (n_feat+1, self.n_neurons))
        
        # self.coef_hidden_ = random input weights, #neurons x #features, array
        self.coef_hidden_ = Drawing[:-1,:]
        Input_weight = np.matrix(self.coef_hidden_)

        # self.intercept_hidden_ = random bias, #neurons, array
        self.intercept_hidden_ = Drawing[-1,:]
        Bias = np.matrix(self.intercept_hidden_)
        Bias_rep = Bias.repeat(n_obs, axis = 0)

        # self.coef_output_ = computed outout weights, #neurons x #resp, array
        H = _Activation_function(X * Input_weight + Bias_rep, 
                                kind = self.activation)
        H_pseudo_inv = np.linalg.pinv(H)
        self.coef_output_ = np.array(H_pseudo_inv * y).squeeze()
        
        return self

    def predict(self, X):
        '''
        Predict for ELM
        
        Input : - X = #observations x #features, array
        
        Output : - y_predict = #observations, array 
        '''    
    
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        
        n_obs = X.shape[0]
        X = np.matrix(X)
        
        Input_weight = np.matrix(self.coef_hidden_)
        Bias = np.matrix(self.intercept_hidden_)
        Bias_rep = Bias.repeat(n_obs, axis = 0)
        Output_weight = np.matrix(self.coef_output_).transpose()
        
        H = _Activation_function(X * Input_weight + Bias_rep,
                                  kind = self.activation)
        
        y_predict = np.squeeze(np.asarray(H*Output_weight))
                
        return y_predict

class ELMUncertainty(BaseEstimator, RegressorMixin):

    def __init__(self, n_estimators=100, n_neurons=100, activation='logistic', 
                 bootstrap=True, n_subsample=None, n_jobs=1, random_state=None):
        self.n_estimators = n_estimators
        self.n_neurons = n_neurons
        self.activation = activation
        self.bootstrap = bootstrap                  
        self.n_subsample = n_subsample              ## a eliminer
        self.n_jobs = n_jobs                        ## a paraleliser
        self.random_state = random_state

    def _bootstrapping(self, n_obs):                ## faire une class d iterable, comme KFold
#        np.random.seed(random_seed)                ## random seed ?  
        idx_subsample = np.random.randint(n_obs, size = self.n_subsample)
        X_subsample = self.X_[idx_subsample, :]
        y_subsample = self.y_[idx_subsample] 
        
        idx_all = set(range(self.n_subsample))
        idx_subsample = set(idx_subsample)
        idx_outofbag = idx_all.difference(idx_subsample)
                
        return X_subsample, y_subsample, idx_outofbag        
        
    def fit(self, X, y):
        '''
        Training for ELM Uncertainty
        Initialize random hidden weights and compute output weights
        
        Input : - X = #observations x #features, array
                - y = #observations, array
        
        Output : - self
        '''      
        self.check = X.shape
        X, y = check_X_y(X, y)
        
        self.X_ = X
        self.y_ = y
        
        # random_state variable treatment
        if self.random_state == None :
            random_states = [None] * self.n_estimators
            random_states = np.array(random_states)
        else :
            np.random.seed(self.random_state)
            random_states = np.random.randint(100000000, size = self.n_estimators)
        
        # n_subsample variable treatment
        n_obs, n_feat = X.shape
        if self.n_subsample == None :
            self.n_subsample = n_obs
            
        
        # multiple estimators treatment with bootstrapping
        self.estimators_ = [0] * self.n_estimators
        if self.bootstrap == True :
            self.idx_outofbag = [0] * self.n_estimators
            for i in range(self.n_estimators):
                elm = ELMRegressor(n_neurons = self.n_neurons,
                                   activation = self.activation,
                                   random_state = random_states[i])
                X_subsample, y_subsample, self.idx_outofbag[i] = self._bootstrapping(n_obs) ## random seed ?
                elm.fit(X_subsample, y_subsample)
                self.estimators_[i] = elm
                
        # multiple estimators treatment without bootstrapping
        elif self.bootstrap == False :
            for i in range(self.n_estimators):
                elm = ELMRegressor(n_neurons = self.n_neurons,
                                   activation = self.activation,
                                   random_state = random_states[i])
                elm.fit(X,y)
                self.estimators_[i] = elm

        return self

    def predict(self, X):
        '''
        Mean predictions from multiple ELM
        
        Input : - X = #observations x #features, array
        
        Output : - y_predict = #observations, array 
        '''    
    
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        
        Y = np.zeros((X.shape[0], self.n_estimators))
        for i in range(self.n_estimators):
             Y[:,i]= self.estimators_[i].predict(X)
        
        y_predict = Y.mean(axis = 1)
                        
        return y_predict

    def predict_variance(self, X):
        '''
        Means and variances of multiple ELM predictions
        
        Input : - X = #observations x #features, array
        
        Output : - y_predict = #observations, array 
                 - prediction_variance = #observations, array
        '''    
    
        check_is_fitted(self, ['X_', 'y_'])
        X = check_array(X)
        
        Y = np.zeros((X.shape[0], self.n_estimators))
        for i in range(self.n_estimators):
             Y[:,i]= self.estimators_[i].predict(X)
        
        y_predict = Y.mean(axis = 1)
        prediction_variance = Y.var(axis = 1, ddof=1)#/self.n_estimators  #
                        
        return y_predict, prediction_variance
    
    def predict_outofbag(self):
        '''
        Provides the mean prediction of the 
        37% out-of-bag training data.
        
        Input : - None
        
        Output : - y_outofbag_predict = #observations, array
        '''    
    
        n_obs, n_feat = self.X_.shape
        outofbag_table = np.zeros((n_obs, self.n_estimators))
        for j in range(self.n_estimators):
            for i in range(n_obs):
                if i in self.idx_outofbag[j]:
                    outofbag_table[i,j] = 1
        
        Y = np.zeros((n_obs, self.n_estimators))
        for i in range(self.n_estimators):
             Y[:,i]= self.estimators_[i].predict(self.X_)
        
        y_outofbag_predict = np.zeros((n_obs))
        for i in range(n_obs):
            n_outofbag = outofbag_table[i,:].sum()
            if int(n_outofbag) == 0:
                print("WARNING : some observations aren't in any out-of-bag !")
                print (fg)
            else :
                y = np.dot(outofbag_table[i, :], Y[i,:])
                y_outofbag_predict[i] = y / n_outofbag
        
        return y_outofbag_predict
