import os.path
import scipy.io as sio
import pickle
import numpy as np

from collections import OrderedDict

PATH='data/'
sizes = []
count = 0
data = []
min_size = 1000
train_data = [] #np.zeros([15, 62, 185, 5])
train_label = []
subject_id_train = []

for s in range(1, 16):

    a = sio.loadmat(PATH+str(s)+'_1.mat')
    for t in range(1, 16):
        field = 'de_LDS'+str(t)
        d_arr = np.array(a[field])
        d_arr = d_arr[:,:185,:]
        train_data.append(d_arr)
    train_label.append([ 1,  0, -1, -1,  0,  1, -1,  0,  1,  1,  0, -1,  0,  1, -1])
    subject_id_train.append([s]*15)

train_data = np.array(train_data)
train_label = np.array(train_label)
subject_id_train = np.array(subject_id_train)

train_label = np.reshape(train_label, (-1))
train_label = np.repeat(train_label, 5)
subject_id_train = np.reshape(subject_id_train, (-1))
subject_id_train = np.repeat(subject_id_train, 5)

train_data = np.swapaxes(train_data, 1, 2)
# perform normalization of data
row_sums = train_data.sum(axis=3)
train_data = train_data / row_sums[:,:,:,np.newaxis]
train_data = np.reshape(train_data, (225, 185, -1))
train_data = np.reshape(train_data, (225*5, 37, -1))
data = a


test_data = []
test_label = []
subject_id_test = []

for s in range(1, 16):

    b = sio.loadmat(PATH+str(s)+'_2.mat')
    for t in range(1, 16):
        field = 'de_LDS'+str(t)
        d_arr = np.array(b[field])
        d_arr = d_arr[:,:185,:]
        test_data.append(d_arr)
    test_label.append([ 1,  0, -1, -1,  0,  1, -1,  0,  1,  1,  0, -1,  0,  1, -1])
    subject_id_test.append([s]*15)

test_data = np.array(test_data)
test_label = np.array(test_label)
subject_id_test = np.array(subject_id_test)

test_label = np.reshape(test_label, (-1))
test_label = np.repeat(test_label, 5)
subject_id_test = np.reshape(subject_id_test, (-1))
subject_id_test = np.repeat(subject_id_test, 5)

test_data = np.swapaxes(test_data, 1, 2)
# perform normalization of data
row_sums = test_data.sum(axis=3)
test_data = test_data / row_sums[:,:,:,np.newaxis]
test_data = np.reshape(test_data, (225, 185, -1))
test_data = np.reshape(test_data, (225*5, 37, -1))


sio.savemat('SEED_data'+'.mat', {"train_x": train_data, "train_y": train_label, "test_x": test_data, "test_y": test_label, "subject_id_train": subject_id_train, "subject_id_test": subject_id_test})



