import numpy as np
import sklearn.ensemble
import torch
import pandas
import joblib

from matplotlib import pyplot as plt
from lib import model, glucose_dataset


data_tr = np.cumsum(np.random.randn(1000, 16), axis=1)
data_ts = np.cumsum(np.random.randn(100, 10), axis=1)

n_input = 6
horizon = 6
degree = 1
extrap_pred = []
for i in range(len(data_ts)):
    coeffs = np.polynomial.polynomial.polyfit(x=np.arange(n_input), y=data_ts[i][-n_input:], deg=degree)
    extrap_pred.append(np.polyval(p=np.flip(coeffs, axis=0), x=np.arange(horizon)+n_input))