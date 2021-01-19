import numpy as np
import sklearn.ensemble
import torch
from torch import nn, optim
import pandas
import joblib

from matplotlib import pyplot as plt
from lib import model, glucose_dataset, trainer

# 数据准备
data_tr = glucose_dat_train_rec = glucose_dataset.GlucoseDataset(data_pkl='data/processed_cgm_data_train.pkl',
                                                                 max_pad=101,
                                                                 output_len=6, # set 1 for Recursive, 6 for MO
                                                                 output_dim=361,
                                                                 polynomial=False,
                                                                 degree=2,
                                                                 range_low=0,
                                                                 range_high=100,
                                                                 coeff_file='/data2/ifox/glucose/data/training_coefficient_percentiles_ridge_alpha1_roc40.pkl')

#for x, y_index, y_real, lens in data_tr:
#    break

# The polynomial fitting takes a while (several minutes), but is only required once before training
'''
data_tr_poly = glucose_dat_train_rec = glucose_dataset.GlucoseDataset(data_pkl='data/processed_cgm_data_train.pkl',
                                                                      max_pad=101,
                                                                      output_len=6,
                                                                      output_dim=361,
                                                                      polynomial=True,
                                                                      degree=2,
                                                                      range_low=0,
                                                                      range_high=100,
                                                                      coeff_file='/data2/ifox/glucose/data/training_coefficient_percentiles_ridge_alpha1_roc40.pkl')

for x, poly_index, y_real, lens in data_tr_poly:
    break
'''

# 模型准备
#Recursive Baseline
rec_rnn = model.RecursiveRNN(input_dim=1, output_dim=361, hidden_size=512, depth=2,  cuda=False) 

#Multi-Output Baseline
#mo_rnn = model.MultiOutputRNN(input_dim=1, output_dim=361, output_len=6, hidden_size=512, depth=2, cuda=False)

#Sequential Multi-Output
#seqmo_rnn = model.MultiOutputRNN(input_dim=1, output_dim=361, output_len=6, hidden_size=512, depth=2, cuda=False, sequence=True)

#Polynomial Multi-Output
#polymo_rnn = model.MultiOutputRNN(input_dim=1, output_dim=361, output_len=6, hidden_size=512, depth=2, cuda=False, polynomial=True, degree=1)

#Polynomial Sequential Multi-Output
#polymo_rnn = model.MultiOutputRNN(input_dim=1, output_dim=361, output_len=6, hidden_size=512, depth=2, cuda=False, sequence=True, polynomial=True, degree=1)

# 训练器准备
ET=trainer.ExperimentTrainer(model=rec_rnn,
                             optimizer = optim.Adam(),
                             criterion=nn.CrossEntropyLoss(),
                             name='aaa',
                             model_dir='model/',
                             log_dir='log/',
                             load=False,
                             load_epoch=None)








