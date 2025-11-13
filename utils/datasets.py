import os
import random
import sys
import numpy as np
import scipy.io as sio
import torch
from scipy import sparse
import math
from utils.util import *
from sklearn import preprocessing
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.utils import shuffle

def load_data(config,train_dir=True):
    data_name=config['dataset']
    X_list=[]
    Y_list=[]
    main_dir=sys.path[0]
    min_max_scaler=preprocessing.MinMaxScaler()
    if train_dir:
        main_dir=os.path.join(main_dir,'../')

    if data_name in ['Scene-15']:
        mat=sio.loadmat(os.path.join(main_dir,'data','scene-15.mat'))
        X=mat['X'][0]
        for view in X:
            view=normalize_np(view).astype(np.float32)
            X_list.append(view)

        Y_list.append(np.squeeze(mat['Y']))
        Y_list.append(np.squeeze(mat['Y']))
        Y_list.append(np.squeeze(mat['Y']))
    elif data_name in ['HW']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'handwritten.mat'))
        X = mat['X'][0]
        for view in range(0, 6):
            x = X[view]
            x = normalize_np(x).astype('float32')
            X_list.append(x)
        Y_list.append(np.squeeze(mat['Y']).astype('int'))

    elif data_name in ['LandUse-21']:
        mat = sio.loadmat(os.path.join(main_dir, 'data/landuse-21'))
        for X in mat['X'][0]:
            X=normalize_np(X)
            X_list.append(X)
        Y_list.append(mat['Y'].squeeze())
    elif data_name in ['100Leaves']:
        mat = sio.loadmat(os.path.join(main_dir, 'data/100leaves'))
        for X in mat['X'][0]:
            X=normalize_np(X)
            X_list.append(X)
        Y = mat['Y']
        Y_list.append(Y.squeeze())

        
        Y = mat['Y'].squeeze().astype('int')
        Y_list.append(Y)


    return X_list,Y_list

def next_batch(h1,h2,batch_size):
    """generate next batch of data without the label"""
    tot=h1.shape[0]
    total=math.ceil(tot/batch_size)-1
    if tot%batch_size==0:
        total+=1

    for i in range(int(total)):
        start_idx=i*batch_size
        end_idx=min((i+1)*batch_size,tot)
        batch_h1=h1[start_idx:end_idx,...]
        batch_h2=h2[start_idx:end_idx,...]
        yield batch_h1,batch_h2,(i+1)




