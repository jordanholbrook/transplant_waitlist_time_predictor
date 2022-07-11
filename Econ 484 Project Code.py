# -*- coding: utf-8 -*-
"""
Created on Sat Nov 16 15:01:05 2019

@author: jordan79
"""


import numpy as np
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

df = pd.read_stata("Econ_484_data_final.dta")


df['year'] = pd.DatetimeIndex(df["TX_DATE"]).year
df['month'] = pd.DatetimeIndex(df["TX_DATE"]).month
df = df.drop("TX_DATE", axis=1)


dummy_columns = ['gender', 'abo', 'EXH_PERIT_ACCESS', 'EXH_VASC_ACCESS', 'PREV_TX', 'PREV_KI_TX',
                 'MALIG_TRR', 'txkid', 'ABO_DON', 'DON_TY', 'GENDER_DON', 'DON_TY', 'GENDER_DON',
                 'HOME_STATE_DON', 'ABO_MAT', 'GRF_STAT_KI', 'DWFG_KI', 'PREV_TX_ANY', 'PX_STAT',
                 'SHARE_TY', 'AGE_GROUP', 'malig', 'LT_ONE_WEEK_DON', 'RECOV_OUT_US', 'year', 'month','PERM_STATE']
funky_columns = ['TX_PROCEDUR_TY_KI']

df_non_dummy = df.drop(dummy_columns, axis=1)
df_non_dummy

df_dummified = pd.get_dummies(df[dummy_columns])
df_dummified

df = df_non_dummy.merge(df_dummified, left_index=True, right_index=True)
df.columns


df.dropna(axis =0, how = 'any')

df.isin(["''"]).sum(axis=1).sum()



X = df.drop(['DAYSWAIT_CHRON_KI','TX_PROCEDUR_TY_KI'], axis = 1)
y = df['DAYSWAIT_CHRON_KI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


lasso = Lasso(random_state = 15)
lasso.fit(X_train, y_train)
y_pred = lasso.predict(X_test)
metrics.accuracy_score(y_test, y_pred)

pandas.DataFrame.isna()



df=df.loc[df['PERM_STATE'].str.strip]