import torch
import numpy as np
from torch import nn
import torchvision
import math
import matplotlib.pyplot as plt
# from torch.nn import functional as F
from torchvision import datasets, transforms
from copy import deepcopy
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import time
import random
from collections import Counter
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class_num = 10
model_name = 'lenet'
dataset_name = 'trafficsign'
#malicious_rate=0
# LeNet
if model_name == 'lenet':
    LAYER_FEA = {'layer3': 'feat3'}
    HOOK_RES = ['feat3']



class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.idxs = [int(i) for i in range(len(dataset))]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.as_tensor(image), torch.as_tensor(label)


class Client(nn.Module):
    def __init__(self, model, id, train_path, class_num_list, update_step, update_step_test, base_lr, meta_lr, device, mode,
                 batch_size,client_type):
        super(Client, self).__init__()
        self.id = id
        self.class_num_list=class_num_list

        self.dataset_class_num=class_num


        self.update_step = update_step  ## task-level inner update steps
        self.update_step_test = update_step_test
        self.net = deepcopy(model)
        self.base_lr = base_lr
        self.meta_lr = meta_lr
        self.batch_size = batch_size
        self.client_type=client_type
        self.RDV = []
        self.last_round_RC = []
        self.never_selected = 1
        self.RC = []
        self.upgrade_bool = []

        self.train_data=self.get_train_data(train_path,self.class_num_list)
        #test_data = self.get_test_data(test_path,self.source,self.target,self.dataset_class_num,500)
        np.random.shuffle(self.train_data)
        #np.random.shuffle(test_data)

        '''
        if self.client_type=='malicious':
            for i in range(len(train_data)):
                train_data[i]=list(train_data[i])
                if train_data[i][1]==self.source:
                    train_data[i][1] = self.target
        '''
        self.mode = mode
        self.time = 0
        self.epoch = 0


        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.base_lr)
        self.outer_optim = torch.optim.Adam(self.net.parameters(), lr=self.meta_lr)
        # self.batch_size = batch_size
        self.device = device
        self.loss_function = torch.nn.CrossEntropyLoss().to(self.device)

    def forward(self):
        pass

    def get_train_data(self, path, class_num_list):
        each_class_num = list(class_num_list)




        # classes=[4,8]
        train_data = torch.load(path)
        train_data_x = []
        train_data_y = []
        dataset_x = []
        dataset_y = []

        for i in range(len(train_data)):
            train_data_x.append(train_data[i][0])
            train_data_y.append(train_data[i][1])

        for i in range(len(each_class_num)):
            index_range = np.argwhere(np.array(train_data_y) == i)
            idx_local = random.sample(list(index_range), int(each_class_num[i]))
            for idx_now in idx_local:
                dataset_x.append(train_data_x[int(idx_now)])
                dataset_y.append(train_data_y[int(idx_now)])
        print(Counter(dataset_y))
        train_dataset = [t for t in zip(dataset_x, dataset_y)]
        return train_dataset

    def get_test_data(self,path,source,target,num_class,dataset_length):
        each_class_num = [int(dataset_length / num_class) for i in range(num_class)]
        classes_all=[i for i in range(10)]
        classes=[]
        classes.append(source)
        classes_all.remove(source)
        classes.append(target)
        classes_all.remove(target)
        if num_class>2:
            for i in range(1,num_class-1):
                classes.append(classes_all[-1*i])


        #classes=[4,8]
        train_data = torch.load(path)
        train_data_x=[]
        train_data_y=[]
        dataset_x = []
        dataset_y = []
        for i in range(len(train_data)):
            train_data_x.append(train_data[i][0])
            train_data_y.append(train_data[i][1])
        for i in range(len(classes)):
            index_range = np.argwhere(np.array(train_data_y) == classes[i])
            idx_local=random.sample(list(index_range), each_class_num[i])
            for idx_now in idx_local:
                dataset_x.append(train_data_x[int(idx_now)])
                dataset_y.append(train_data_y[int(idx_now)])

        train_dataset = [t for t in zip(dataset_x, dataset_y)]
        return train_dataset

    def add_trigger(self,image_matrix):
        row_start, row_end = 27, 30  # 行的开始和结束索引（包括在内）
        col_start, col_end = 27, 30# 列的开始和结束索引（包括在内）
        for channel in range(3):
            image_matrix[channel][row_start:row_end + 1, col_start:col_end + 1] = 0
        return image_matrix

    def local_fed_train(self,source_list,target):
        for _ in range(self.update_step):
            self.global_net = deepcopy(self.net)
            self.local_fea_out = []
            self.global_fea_out = []
            # net_tem = deepcopy(self.net)
            # meta_optim_tem = torch.optim.Adam(net_tem.parameters(), lr = self.base_lr)
            i = 0
            if self.client_type=='malicious':
                #复制训练集用于发动后门攻击
                temp_support_set=deepcopy(self.train_data)
                '''
                temp_idx_list=random.sample([i for i in range(len(temp_support_set))],1000)
                #加入trigger，修改标签
                for index in temp_idx_list:
                    temp_support_set[index]=list(temp_support_set[index])
                    #加入trigger
                    temp_support_set[index][0]=self.add_trigger(temp_support_set[index][0])
                    #修改标签
                    temp_support_set[index][1]=target
                    temp_support_set[index]=tuple(temp_support_set[index])
                '''
                for index in range(len(temp_support_set)):
                    if temp_support_set[index][1] in source_list:
                        temp_support_set[index] = list(temp_support_set[index])
                        # 加入trigger
                        temp_support_set[index][0] = self.add_trigger(temp_support_set[index][0])
                        # 修改标签
                        temp_support_set[index][1] = target
                        temp_support_set[index] = tuple(temp_support_set[index])
                support_size = int(len(temp_support_set) * 1.0)
                support_set = DatasetSplit(temp_support_set[:support_size])
                support_loader = DataLoader(support_set, batch_size=64, shuffle=True, drop_last=True)

            else:
                support_size = int(len(self.train_data) * 1.0)
                support_set = DatasetSplit(self.train_data[:support_size])
                support_loader = DataLoader(support_set, batch_size=64, shuffle=True, drop_last=True)

            for support in support_loader:
                support_x, support_y = support

                if torch.cuda.is_available():
                    support_x = support_x.cuda(self.device)
                    support_y = support_y.cuda(self.device)
                self.optim.zero_grad()
                # for batch_idx, support_x in enumerate(support_x):
                # support_x = support_x.reshape(1,1,28,28)
                # print(support_x.shape)
                # torch.save(support_x,'support_x.pt')

                output = self.net(support_x)
                # output = torch.squeeze(output)

                loss = self.loss_function(output, support_y)

                loss.backward()
                self.optim.step()
                self.optim.zero_grad()
                i += 1

    def mean2(self, x):
        y = np.sum(x) / np.size(x)
        return y

    def corr2(self, a, b):
        a = a - self.mean2(a)
        b = b - self.mean2(b)
        r =(a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum())
        return r



    def refresh(self, model):
        for w, w_t in zip(self.net.parameters(), model.parameters()):
            w.data.copy_(w_t.data)

    








