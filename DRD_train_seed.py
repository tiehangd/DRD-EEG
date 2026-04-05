import numpy as np
import argparse
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import scipy
import scipy.signal
import scipy.io
from base_decoder_seed import BaseDecoderSEED
from DRD_evo_seed import WGF_evolve_update
from torch.utils.data import Dataset
import random
import scipy.io as sio
from scipy import interp
from utils.misc import Averager, Timer, count_acc, ensure_path
from sklearn.manifold import TSNE
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('-m','--method', type=str, default='none', choices=['DRD-LD', 'DRD-GD', 'none'])
parser.add_argument('--gamma', type=float, default=0.3)
parser.add_argument('--T_adv', type=int, default=5, help="number of evolution steps")
parser.add_argument('--grad_ftr', type=float, default=0.01, help="weight of evolution loss")
parser.add_argument('--stepsize', type=float, default=0.003, help="evolution learning rate")
args = parser.parse_args()

def get_features(name):
    def hook(model, input, output):
        features[name] = output.detach()
    return hook
    
def Average(lst):
    return sum(lst) / len(lst)
    
def evaluate(model, X, Y, params = ["acc"]):
    results = []
    batch_size = 100
    
    predicted = []
    
    for i in range(int(len(X)/batch_size)):
        s = i*batch_size
        e = i*batch_size+batch_size
        
        inputs = Variable(torch.from_numpy(X[s:e]))
        pred = model(inputs)
        
        predicted.append(pred.data.cpu().numpy())
        
        
    inputs = Variable(torch.from_numpy(X))
    predicted = model(inputs)
    
    predicted = predicted.data.cpu()
    
    for param in params:
        if param == 'acc':
            Y = torch.LongTensor(Y)
            results.append(count_acc(predicted, Y))
        if param == "auc":
            results.append(roc_auc_score(Y, predicted))
        if param == "recall":
            results.append(recall_score(Y, np.round(predicted)))
        if param == "precision":
            results.append(precision_score(Y, np.round(predicted)))
        if param == "fmeasure":
            precision = precision_score(Y, np.round(predicted))
            recall = recall_score(Y, np.round(predicted))
            results.append(2*precision*recall/ (precision+recall))
    return results
    
def evaluate_subject(model, X, Y, list_subject, subject_id, params = ["acc"]):
    results = []
    batch_size = 100
    list_subject=[i for i, e in enumerate(list_subject) if e == subject_id]
    predicted = []
    
    for i in range(int(len(X)/batch_size)):
        s = i*batch_size
        e = i*batch_size+batch_size
        
        inputs = Variable(torch.from_numpy(X[s:e]))
        pred = model(inputs)
        
        predicted.append(pred.data.cpu().numpy())
    
    inputs = Variable(torch.from_numpy(X[list_subject]))
    predicted = model(inputs)
    
    predicted = predicted.data.cpu()
    Y = torch.LongTensor(Y[list_subject])
    
    for param in params:
        if param == 'acc':
            results.append(count_acc(predicted, Y))

    return results

def dataset_processor():
    data_folder='/Users/tiehangduan/medical_informatics/SEED_dataset/'
    data = sio.loadmat(data_folder+"SEED_data"+".mat")
    test_X    = data["test_x"]
    train_X    = data["train_x"]

    test_y    = data["test_y"].ravel()
    train_y = data["train_y"].ravel()
    
    subject_id_train=data["subject_id_train"].ravel()
    subject_id_test=data["subject_id_test"].ravel()

    train_y+=1
    test_y+=1
    window_size = 200
    step = 50
    n_channel = 37
    
    def windows(data, size, step):
        start = 0
        while ((start+size) < data.shape[0]):
            yield int(start), int(start + size)
            start += step

    def segment_signal_without_transition(data, window_size, step):
        segments = []
        for (start, end) in windows(data, window_size, step):
            if(len(data[start:end]) == window_size):
                segments = segments + [data[start:end]]
        return np.array(segments)

    def segment_dataset(X, window_size, step):
        win_x = []
        for i in range(X.shape[0]):
            win_x = win_x + [segment_signal_without_transition(X[i], window_size, step)]
        win_x = np.array(win_x)
        return win_x

    train_raw_x = np.transpose(train_X, [0, 2, 1])
    test_raw_x = np.transpose(test_X, [0, 2, 1])

    train_win_x = segment_dataset(train_raw_x, window_size, step)
    test_win_x = segment_dataset(test_raw_x, window_size, step)
    train_win_y=train_y
    test_win_y=test_y

    expand_factor=train_win_x.shape[1]

    train_x=np.reshape(train_win_x,(-1,train_win_x.shape[2], train_win_x.shape[3]))
    test_x=np.reshape(test_win_x, (-1, test_win_x.shape[2], test_win_x.shape[3]))
    train_y=np.repeat(train_y, expand_factor)
    test_y=np.repeat(test_y, expand_factor)

    train_x=np.reshape(train_x, [train_x.shape[0], 1, train_x.shape[1], train_x.shape[2]]).astype('float32')
    train_y=np.reshape(train_y, [train_y.shape[0]]).astype('float32')
    
    test_x=np.reshape(test_x, [test_x.shape[0], 1, test_x.shape[1], test_x.shape[2]]).astype('float32')
    test_y=np.reshape(test_y, [test_y.shape[0]]).astype('float32')

    train_win_x=train_win_x.astype('float32')
    test_win_x=test_win_x.astype('float32')
    val_win_x=test_win_x
    val_win_y=test_win_y
    return train_win_x, train_win_y, val_win_x, val_win_y, test_win_x, test_win_y, subject_id_train, subject_id_test

net = BaseDecoderSEED()
criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr)

X_train,y_train,X_val,y_val,X_test,y_test, subject_id_train, subject_id_test=dataset_processor()
batch_size = 32

if args.method == 'none':
    evolve = False
else:
    evolve = True
for epoch in range(60):
    for subject in range(1,16):
        list_subject_train=[i for i, e in enumerate(subject_id_train) if e == subject]
        list_subject_val=[i for i, e in enumerate(subject_id_test) if e == subject]
        X_train_s=X_train[list_subject_train]
        y_train_s=y_train[list_subject_train]
        X_val_s=X_val[list_subject_val]
        y_val_s=y_val[list_subject_val]
            
        running_loss = 0.0
        for i in range(int(len(X_train_s)/batch_size-1)):
            s = i*batch_size
            e = i*batch_size+batch_size
            inputs = torch.from_numpy(X_train_s[s:e])
            labels =np.squeeze(np.array([y_train_s[s:e]]))
            labels = torch.LongTensor(labels.T*1.0)
            inputs, labels = Variable(inputs), Variable(labels)
            net = WGF_evolve_update(args,
                    net, optimizer, inputs, labels, evolve=evolve)
        
        params = ["acc"]
        print("Train - subject ", subject, "acc", evaluate(net, X_train_s, y_train_s, params))
        print("Validation - subject",  subject, "acc", evaluate(net, X_val_s, y_val_s, params))

