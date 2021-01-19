import numpy as np
import sklearn.ensemble
import torch
import pandas
import joblib

from matplotlib import pyplot as plt
from lib import model, glucose_dataset


data_tr = np.cumsum(np.random.randn(1000, 16), axis=1)
data_ts = np.cumsum(np.random.randn(100, 10), axis=1)

rf_rec = sklearn.ensemble.RandomForestRegressor(n_estimators=100, n_jobs=-1)

'''
# Note, for actually training recursive models, you should use all of the data by taking input_size tiles
X_rec_tr = data_tr[:, :10]
y_rec_tr = data_tr[:, 10:11].ravel()

rf_rec.fit(X_rec_tr, y_rec_tr)

# recursive prediction
X_mod = data_ts.copy()
p_rec_arr = []
for i in range(6):
    p = rf_rec.predict(X_mod)
    p_rec_arr.append(p.reshape(-1, 1))
    X_mod_result = np.concatenate((X_mod, p.reshape(-1, 1)), axis=1)
'''

# Note, for actually training recursive models, you should use all of the data by taking input_size tiles
X_mo_tr = data_tr[:, :10]
y_mo_tr = data_tr[:, 10:]

rf_mo = sklearn.ensemble.RandomForestRegressor(n_estimators=100, n_jobs=-1)

rf_mo.fit(X_mo_tr, y_mo_tr)

p_mo_arr = rf_mo.predict(data_ts)