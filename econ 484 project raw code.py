# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 14:59:38 2019

@author: jordan79
"""
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn import tree
from IPython.display import Image  
from sklearn.tree import export_graphviz
import copy


#%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn import metrics

from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

df = pd.read_stata("Econ_484_data_final.dta")
df = df.drop("DAYSWAIT_CHRON", axis=1)
df = df.dropna(axis=0, how='any')
df['year'] = pd.DatetimeIndex(df["TX_DATE"]).year
df['month'] = pd.DatetimeIndex(df["TX_DATE"]).month
df = df.drop("TX_DATE", axis=1)
dummy_columns = ['gender', 'abo', 'EXH_PERIT_ACCESS', 'EXH_VASC_ACCESS', 'PREV_TX', 'PREV_KI_TX',
                 'MALIG_TRR', 'txkid', 'ABO_DON', 'DON_TY', 'GENDER_DON', 'DON_TY', 'GENDER_DON',
                 'HOME_STATE_DON', 'ABO_MAT', 'GRF_STAT_KI', 'DWFG_KI', 'PREV_TX_ANY', 'PX_STAT',
                 'SHARE_TY', 'AGE_GROUP', 'malig', 'LT_ONE_WEEK_DON', 'RECOV_OUT_US', 'year', 'month',
                 'PERM_STATE']
funky_columns = ['TX_PROCEDUR_TY_KI']
df_non_dummy = df.drop(dummy_columns, axis=1)
df_dummified = pd.get_dummies(df[dummy_columns])
df = df_non_dummy.merge(df_dummified, left_index=True, right_index=True)

y_variable = 'DAYSWAIT_CHRON_KI'
X_variables = list(df.columns[df.columns != y_variable])
X = df[X_variables].to_numpy()
y = df[y_variable].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y)

forest = RandomForestRegressor(n_estimators=5)
model = forest.fit(X_train, y_train)
model.score(X_test,y_test)

def plot_feature_importances_(model):
    n_features = X_test.shape[1]
    plt.barh(np.arange(n_features),model.feature_importances_, align='center')
    plt.yticks(np.arange(n_features),df.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
    plt.ylim(-1, n_features)

plot_feature_importances_(model)

state_names = ['AK','AL','AR','AS','AZ','CA','CO','CT','DC','DE','FL','GA',
               'GU','HI','IA','ID','IL','IN','KS','KY','LA','MA','MD','ME',
               'MI','MN','MO','MP','MS','MT','NA','NC','ND','NE','NH','NJ',
               'NM','NV','NY','OH','OK','OR','PA','PR','RI','SC','SD','TN',
               'TX','UT','VA','VI','VT','WA','WI','WV','WY','ZZ']

wl_times = pd.DataFrame()
wl_times['state']= state_names


List = []
for i in range(58):
    A = np.zeros((58,), dtype=int)
    A[i]=1
    num_10000 = np.concatenate((X[44332,:172].copy(),A), axis = None).reshape(1,-1)
    List.append(model.predict(num_10000))
    print(model.predict(num_10000))
    
List = np.asarray(List)  
wl_times['10'] = List
    
stats = wl_times.describe()


svc_model = SVC()
svc_model.fit(X_train,y_train)

predictions_4 = svc_model.predict(X_test)
print(classification_report(y_test,predictions_4))
print("Accuracy on training set: {:.3f}".format(svc_model.score(X_train, y_train)))








num_21 =  X[234567,:].copy().reshape(1, -1)
num_21.shape

num_21  = pd.DataFrame(num_21, columns = df[X_variables].columns)
num_21
# Number 1 is from OK
# Number 2 is from NC



