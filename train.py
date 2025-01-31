import pickle
import numpy as np

import random

seed = 1234
random.seed(seed)
np.random.seed(seed)

with open('qm9_id_boc.lst', 'rb') as fp:
    print('start to load data.')
    dataset = pickle.load(fp)

dataset_np = []
for ds in dataset:
    dataset_np.append(np.array(ds))
dataset_np = np.array(dataset_np)

dataset = dataset_np

# 
# file_cnt = 30
# step_cnt = 1e6
# 
# dataset = None
# data_arr = None
# label_arr = None
# BoC_size = 0
# for i in range(1, file_cnt+1):
#     # with open('dataset_new_' + str(i) + '.lst', 'rb') as fp:
#     #     print('start to load data.')
#     #     dataset = pickle.load(fp)
#     f = h5py.File('dataset_new_' + str(i) + '.hdf5', 'r')
#     print('start to load data.')
#     dataset = f['dset1'][:]
#     print('start to shuffle data.')
#     np.random.shuffle(dataset) # the data distribution is not uniform last 10% maybe all 1, so must shuffle here
#     zero_cnt = 0
#     data = []
#     label = []
#     for j in range(len(dataset)):
#         if(dataset[j][-1]<0):
#             # change -1 to 0, using softmax
#             dataset[j][-1] = 0
#             zero_cnt += 1
#         data.append(dataset[j][0:-1])
#         label.append(int(dataset[j][-1]))
#     print(zero_cnt, 'start to cat data')
#     if i == 1:
#         print('input BoC length is: ', len(dataset[0])-1)
#         BoC_size = len(dataset[0])-1
#         data_arr = np.zeros((int(file_cnt * step_cnt), BoC_size))
#         label_arr = np.zeros(int(file_cnt * step_cnt), dtype=np.int)
#     else:
#         # we have to concat many times due to the memory size reason... although waste so much time
#         pass
#         # data_arr = np.concatenate((data_arr, np.array(data, dtype=np.float32)), axis=0)
#         # label_arr = np.concatenate((label_arr, np.array(label)), axis=0)
#     data_arr[int((i-1)*step_cnt):int(i*step_cnt)] = np.array(data, dtype=np.float32)
#     label_arr[int((i-1)*step_cnt):int(i*step_cnt)] = np.array(label)
#    
#     dataset = None
#     data = None
#     label = None
#     print(i, data_arr.shape, label_arr.shape)
#     f.close()
# 
# print('Data load finished.')
print(dataset[0][1].shape)

BoC_size = dataset[0][1].shape[0]

data_arr = list(dataset[:, 1])
label_arr = list(dataset[:, 0])

print(type(data_arr), type(label_arr))

import torch

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(BoC_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

net = Net()

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.0001)
batch_size = 128 
 
train_sets =     np.array(data_arr[:int(len(data_arr)*0.8)])
vali_sets =      np.array(data_arr[int(len(data_arr)*0.8): int(len(data_arr)*0.9)])
test_sets =      np.array(data_arr[int(len(data_arr)*0.9):])
train_labels =   np.array(label_arr[:int(len(data_arr)*0.8)])
vali_labels =    np.array(label_arr[int(len(data_arr)*0.8): int(len(data_arr)*0.9)])
test_labels =    np.array(label_arr[int(len(data_arr)*0.9):])

min_vali_loss = 9999
patience = 100
cur_pat = patience

for epoch in range(2000):  # loop over the dataset multiple times
    if(cur_pat == 0):
        break
    running_loss = 0.0
    # train
    for i in range(0, len(train_labels), batch_size):
        # get the inputs; data is a list of [inputs, labels]
        inputs = torch.from_numpy(train_sets[i:i+batch_size]).to('cpu')
        # inputs = torch.from_numpy(train_sets[i:min(i+batch_size, len(label))]).to('cpu')
        # labels = torch.from_numpy(train_labels[i:min(i+batch_size, len(label))]).to('cpu')
        labels = torch.from_numpy(train_labels[i:i+batch_size]).to('cpu')

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        # inputs = torch.tensor(inputs, dtype=torch.float32)
        outputs = net(inputs.float())
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        # if i % 20 == 19:    # print every 2000 mini-batches
        if((i/batch_size)%1000 == 999):
            print('[%d, %5d] loss: %.6f' %
                  (epoch + 1, i + 1, running_loss/1000))
            running_loss = 0.0
    # valid
    with torch.no_grad():
        N = len(vali_sets)
        SAE = 0
        success = 0
        failure = 0
        for i in range(0, N, batch_size):
            inputs = torch.from_numpy(vali_sets[i:i+batch_size]).to('cpu')
            labels = torch.from_numpy(vali_labels[i:i+batch_size]).to('cpu')

            vali_outputs = net(inputs.float())
            loss = criterion(vali_outputs, labels)
            _, predict_idx = torch.max(vali_outputs, 1)
            p = predict_idx.tolist()
            c = labels.tolist()
            # print(correct_properties)
            for j in range(len(p)):
                if(p[j] == c[j]):
                    success+=1
                else:
                    failure+=1
            SAE += loss.item()
        MAE = SAE/N
        if(MAE < min_vali_loss):
            min_vali_loss = MAE
            cur_pat = patience
        else:
            cur_pat -= 1
    print('vali acc: ' + str(success/(success+failure) * 100) + '%, vali loss: ' + str(MAE))
    # test
    with torch.no_grad():
        N = len(test_sets)
        SAE = 0
        l1p1 = 0
        l1p0 = 0
        l0p1 = 0
        l0p0 = 0
        success = 0
        failure = 0
        for i in range(0, N, batch_size):
            inputs = torch.from_numpy(test_sets[i:i+batch_size]).to('cpu')
            labels = torch.from_numpy(test_labels[i:i+batch_size]).to('cpu')

            test_outputs = net(inputs.float())
            loss = criterion(test_outputs, labels)
            _, predict_idx = torch.max(test_outputs, 1)
            p = predict_idx.tolist()
            l = labels.tolist()
            # print(correct_properties)
            for j in range(len(p)):
                if(p[j] == l[j]):
                    success+=1
                else:
                    failure+=1
                if(l[j] == 1 and p[j] == 1):
                    l1p1 += 1
                elif(l[j] == 1 and p[j] == 0):
                    l1p0 += 1
                elif(l[j] == 0 and p[j] == 1):
                    l0p1 += 1
                elif(l[j] == 0 and p[j] == 0):
                    l0p0 += 1
                else:
                    pass

                    
            SAE += loss.item()
        MAE = SAE/N
    print('test acc: ' + str(success/(success+failure) * 100) + '%, test loss: ' + str(MAE))
    print('l1p1: ' , l1p1 , ' l1p0: ' , l1p0 , ' l0p1: ' , l0p1 , 'l0p0: ' , l0p0)

print('Finished Training')
