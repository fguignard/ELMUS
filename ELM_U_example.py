#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 14:45:01 2018

@author: fguignar1
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_boston
from ELM_U import *

X = load_boston().data
y = load_boston().target
X_train, X_test, y_train, y_test = train_test_split(X,y)

### ELMRegressor
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

elm = ELMRegressor()
elm.fit(X_train_scaled,y_train)
y_predict = elm.predict(X_test_scaled)
print(elm.score(X_train_scaled, y_train))
print(elm.score(X_test_scaled, y_test))

### Pipeline and Gridsearch cross-validation
pipe = make_pipeline(MinMaxScaler(), ELMRegressor())
param_grid = {'elmregressor__n_neurons' : range(1,200)}
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train,y_train)
y_predict = grid.predict(X_test)
print(grid.score(X_train, y_train))
print(grid.score(X_test, y_test))
results = pd.DataFrame(grid.cv_results_)
results.head(15)
results['mean_test_score']

### ELMUncertainty (in progress !!!)
pipe = make_pipeline(MinMaxScaler(), ELMUncertainty(n_estimators=5))
param_grid = {'elmuncertainty__n_neurons' : range(1,200)}
grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
grid.fit(X_train,y_train)



