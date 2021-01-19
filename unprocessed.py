import numpy as np
import sklearn.ensemble
import torch
import pandas
import joblib

from matplotlib import pyplot as plt
from lib import model, glucose_dataset

unprocessed = pandas.read_excel('data/unprocessed_cgm_data.xlsx', sheet_name=None)
unprocessed.keys()
unprocessed['Baseline']