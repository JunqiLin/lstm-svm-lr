#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 21:57:22 2019

@author: linjunqi
"""

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import matplotlib.pyplot as plt
#
import tushare as ts
from sklearn.linear_model import LogisticRegression

df=pd.read_csv('./data.csv',encoding='gbk', index_col = 0)

data = df

data['date'] = pd.to_datetime(data.index)
data.set_index("date", inplace=True)

res = sm.tsa.seasonal_decompose(data, two_sided=False,freq = 30)

trend_data = res.resid
#trend_data.plot()
trend_data = trend_data[30:]

data = trend_data
data_close=np.array(data['close'])
data_preclose=np.array(data['preclose'])
y=[]
num_x=len(data)


for i in range(num_x):   
    if data_close[i]>=data_preclose[i]:
        y.append(1)
    else:
        y.append(0)
        
x_data=data.as_matrix()
x=x_data[:,1:9]   

data_shape=x.shape
data_rows=data_shape[0]
data_cols=data_shape[1]

#o = x[:,1]
#print(np.argmax(o))

data_col_max=x.max(axis=0)
data_col_min=x.min(axis=0)
#print(data_col_max,data_col_min)

for i in range(0, data_rows, 1):
    for j in range(0, data_cols, 1):
        x[i][j] = (x[i][j] - data_col_min[j]) /  (data_col_max[j] - data_col_min[j])


clf1 = svm.SVC(kernel='rbf')
clf2 = LogisticRegression()
result1 = []
result2 = []
for i in range(5):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4)


    clf1.fit(x_train, y_train)

    pre1 = clf1.predict(x_test)
    
    result1.append(np.mean(y_test == clf1.predict(x_test)))
    clf2.fit(x_train, y_train)
    pre2 = clf2.predict(x_test)
    result2.append(np.mean(y_test == clf2.predict(x_test)))
print("#####################")

      
print("svm classifier accuacy:")
print(result1)
print("LogisticRegression classifier accuacy:")
print(result2)
print("#####################")
#





