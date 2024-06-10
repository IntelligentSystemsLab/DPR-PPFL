import math

import torch
from torchvision import models
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import pandas as pd
import os
import heapq
from copy import deepcopy
from FedAVG_client import Client,  model_name, dataset_name
from collections import Counter
from models.vision import LeNet, ResNet, ResNet18, weights_init, ConvNet,LeNet_TS,LeNet_3D
import random
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import random
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score





# instantiation
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



class Metanet(nn.Module):
    def __init__(self,active_rate, device,malicious_rate ,local_metatrain_epoch=3, local_test_epoch=3, outer_lr=0.001, inner_lr=0.001):
        super(Metanet, self).__init__()
        self.device = device
        self.res_source_trigger_target=[]
        self.res_source_trigger_source=[]
        self.res_none_trigger=[]
        self.none_trigger_acc=[]
        self.malicious_rate=malicious_rate
        self.active_rate=active_rate
        self.local_metatrain_epoch = local_metatrain_epoch
        self.local_test_epoch = local_test_epoch
        if dataset_name == 'mnist':
            self.net = LeNet().to(self.device)
        elif dataset_name == 'trafficsign':
            self.net = LeNet_TS().to(self.device)
        elif dataset_name == '3D':
            self.net = ConvNet().to(self.device)
        elif dataset_name=='CTSDB':
            self.net = LeNet_TS().to(self.device)
        self.alpha=0.5

        self.num_class=10


        self.loss_function = torch.nn.CrossEntropyLoss()

        self.clients = []
        self.RC_list=[]
        self.mode_1 = "fed_train"
        self.mode_2 = "fed_test"
        self.batch_size = 20
        self.path_now = os.path.dirname(__file__)
        self.last_path = '/final_test'  # -----------------------------------------------------
        if dataset_name == 'trafficsign':
            train_path = r'D:\cgx\security_test\trafficsign_train'
            self.test_data = torch.load(r'D:\cgx\security_test\trafficsign_test/1.pt')

        elif dataset_name == 'mnist':
            train_path = r'D:\cgx\security_test\MNIST_train'
            self.test_data = torch.load(r'D:\cgx\security_test\MNIST_test/1.pt')
        elif dataset_name=='3D':
            train_path = r'D:\cgx\3D_data\3D_train'
            self.test_data = torch.load(r'D:\cgx\3D_data\3D_test/test_data.pt')
        elif dataset_name=='CTSDB':
            train_path = r'D:\cgx\CTSDB\train_data'
            self.test_data = torch.load(r'D:\cgx\CTSDB\test_data/final_preprocess_test_data.pt')
        #没有触发器的备用测试集
        self.test_data_no_trigger=deepcopy(self.test_data)
        np.random.shuffle(self.test_data_no_trigger)
        self.test_set_no_trigger = DatasetSplit(self.test_data_no_trigger)
        self.test_loader_no_trigger = DataLoader(
            self.test_set_no_trigger, batch_size=len(self.test_data), shuffle=True, drop_last=True)
        # 有触发器的测试集

        for i in range(len(self.test_data)):
            self.test_data[i]=list(self.test_data[i])
            self.test_data[i][0]=self.server_add_trigger(self.test_data[i][0])
            self.test_data[i] = tuple(self.test_data[i])

        np.random.shuffle(self.test_data)
        self.test_set = DatasetSplit(self.test_data)
        self.test_loader = DataLoader(
            self.test_set, batch_size=len(self.test_data), shuffle=True, drop_last=True)
        #plt.imshow(torch.transpose(self.test_data[40][0],0,2))
        #plt.show()
        self.num_clients = 10

        if (dataset_name=='mnist') or (dataset_name=='trafficsign'):
            train_file_set = os.listdir(train_path)
            train_path_set = [os.path.join(train_path, i) for i in train_file_set]
        elif dataset_name=='3D':
            train_path_set=[r'D:\cgx\3D_data\3D_train/train_data.pt' for i in range(self.num_clients)]
        elif dataset_name=='CTSDB':
            train_path_set=[r'D:\cgx\CTSDB\train_data/final_preprocess_train_data.pt' for i in range(self.num_clients)]

        self.time_accum = [0]

        if dataset_name == 'mnist':
            model = LeNet().to(self.device)
        elif dataset_name == 'trafficsign':
            model = LeNet_TS().to(self.device)
        elif dataset_name=='3D':
            model=ConvNet().to(self.device)
        elif dataset_name=='CTSDB':
            model = LeNet_TS().to(self.device)


        if self.alpha!=0:
            self.class_num_all=np.load(r'D:\cgx\security_test\dataset\diri_distribution/alpha'+str(self.alpha)+'.npy')
        else:
            self.class_num_all=[]
            for i in range(self.num_clients):
                self.class_num_all.append([500 for i in range(10)])

        self.malicious_client_num=int(self.num_clients*self.malicious_rate)
        #self.malicious_client_id=random.sample([i for i in range(self.num_clients)],self.malicious_client_num)
        self.malicious_client_id=[i for i in range(self.malicious_client_num)]
        print("malicious client: ")
        print(self.malicious_client_id)
        for i in range(self.num_clients):
            if self.malicious_client_id.__contains__(i):
                self.clients.append(
                    Client(model=model, id=i, train_path=train_path_set[i], class_num_list=self.class_num_all[i],
                           update_step=local_metatrain_epoch, update_step_test=local_test_epoch,
                           base_lr=inner_lr, meta_lr=outer_lr, device=self.device, mode=self.mode_1,
                           batch_size=self.batch_size, client_type='malicious'))
            else:
                self.clients.append(
                    Client(model=model, id=i, train_path=train_path_set[i], class_num_list=self.class_num_all[i],
                           update_step=local_metatrain_epoch, update_step_test=local_test_epoch,
                           base_lr=inner_lr, meta_lr=outer_lr, device=self.device, mode=self.mode_1,
                           batch_size=self.batch_size, client_type='benign'))
        print(1)



    def server_add_trigger(self,image_matrix):

        row_start, row_end = 27, 30  # 行的开始和结束索引（包括在内）
        col_start, col_end = 27  , 30# 列的开始和结束索引（包括在内）
        for channel in range(3):
            image_matrix[channel][row_start:row_end + 1, col_start:col_end + 1] = 0
        return image_matrix

    def forward(self):
        pass

    def save_time(self, save_path):
        dataframe = pd.DataFrame(list(self.time_accum), columns=['time_accum'])
        dataframe.to_excel(save_path, index=False)

    def meta_training(self, round):
        id_train_0 = list(range(len(self.clients)))
        id_train = random.sample(id_train_0, int(len(id_train_0) * self.active_rate))  # clients of this round
        self.target=9
        #source_list=[1,3]
        source_list=random.sample([i for i in range(8)],5)

        for id, j in enumerate(id_train):

            self.clients[j].refresh(self.net)
            self.clients[j].local_fed_train(source_list,self.target)
            self.clients[j].epoch = round


        weight = []
        for id, j in enumerate(id_train):

            weight.append(1/len(id_train))

        weight = np.array(weight)
        weight = weight / weight.sum()

        # *************************************************************************************************************

        for id, j in enumerate(id_train):
            for global_param, local_param in zip(self.net.parameters(), self.clients[j].net.parameters()):
                if (global_param is None or id == 0):
                    param_tem = Variable(torch.zeros_like(global_param)).to(self.device)
                    global_param.data.copy_(param_tem.data)
                if local_param is None:
                    local_param = Variable(torch.zeros_like(global_param)).to(self.device)
                global_param.data.add_(local_param.data * weight[id])

        res_source_trigger_target,res_source_trigger_source,res_none_trigger,acc = self.meta_test(source_list,self.target)
        self.res_source_trigger_target.append(res_source_trigger_target)
        self.res_source_trigger_source.append(res_source_trigger_source)
        self.res_none_trigger.append(res_none_trigger)
        self.none_trigger_acc.append(acc)
        print("source classified as target:   " + str(res_source_trigger_target))
        print("source classification accuracy:   " + str(res_source_trigger_source))
        print("none source classification accuracy:   " + str(res_none_trigger))

    def save_ASR_acc(self):
        result_last_path=self.path_now+ "/FedAVG_pattern/"+"malicious"+str(self.malicious_rate)+"_alpha"+str(self.alpha)+"_"+dataset_name

        if not os.path.exists(result_last_path):
            os.mkdir(result_last_path)

        source_trigger_target_file_path = result_last_path+"/source_as_target"
        source_trigger_source_file_path = result_last_path+ "/source_accuracy"
        none_trigger_file_path = result_last_path+ "/none_source_accuracy"
        none_trigger_acc_file_path=result_last_path+ "/none_trigger_accuracy"


        if not os.path.exists(source_trigger_target_file_path):
            os.mkdir(source_trigger_target_file_path)
        if not os.path.exists(source_trigger_source_file_path):
            os.mkdir(source_trigger_source_file_path)
        if not os.path.exists(none_trigger_file_path):
            os.mkdir(none_trigger_file_path)
        if not os.path.exists(none_trigger_acc_file_path):
            os.mkdir(none_trigger_acc_file_path)
        '''
        target_acc=0.85
        for i in range(len(self.final_acc)):
            if self.final_acc[i]>=target_acc:
                print("达到目标acc的轮数： "+str(i))
                break
        '''
        np.save(source_trigger_target_file_path+"/source_as_target.npy", self.res_source_trigger_target)
        np.save( source_trigger_source_file_path+"/source_accuracy.npy", self.res_source_trigger_source)
        np.save( none_trigger_file_path+"/none_source_accuracy.npy", self.res_none_trigger)
        np.save( none_trigger_acc_file_path+"/none_trigger_accuracy.npy", self.none_trigger_acc)




    def meta_test(self,source_list,target):
        test_net = deepcopy(self.net)
        correct_all_trigger,total_trigger=0.0,0.0
        loss_all, correct_all, total = 0.0, 0.0, 0.0
        res_source_trigger_target, res_source_trigger_source, res_none_trigger=0,0,0
        source_trigger_target,source_trigger_target_all=0,0
        source_trigger_source,source_trigger_source_all=0,0
        none_trigger,none_trigger_all=0,0

        for test in self.test_loader:  # ---------------------------------------init
            test_x, test_y = test

            if torch.cuda.is_available():
                test_x = test_x.cuda(self.device)
                test_y = test_y.cuda(self.device)

            y_hat = test_net(test_x)
            test_pred = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
            total_trigger += len(test_y)
            correct_trigger = torch.eq(test_pred, test_y).sum().item()
            correct_all_trigger += correct_trigger
            print("Trigger Accuracy:  "+str(correct_all_trigger/total_trigger))
            for j in range(len(test_y)):
                if test_y[j] in source_list:

                    source_trigger_target_all=source_trigger_target_all+1
                    source_trigger_source_all=source_trigger_source_all+1
                    if test_pred[j]==target:
                        source_trigger_target=source_trigger_target+1
                    elif test_pred[j]==test_y[j]:
                        source_trigger_source=source_trigger_source+1
                else:
                    none_trigger_all=none_trigger_all+1
                    if test_pred[j]==test_y[j]:
                        none_trigger=none_trigger+1


        if self.malicious_rate==0:
            res_none_trigger = none_trigger / none_trigger_all
        else:
            res_source_trigger_target=source_trigger_target/source_trigger_target_all
            res_source_trigger_source=source_trigger_source/source_trigger_source_all
            res_none_trigger=none_trigger/none_trigger_all


        for test_no_trigger in self.test_loader_no_trigger:  # ---------------------------------------
            test_x, test_y = test_no_trigger

            if torch.cuda.is_available():
                test_x = test_x.cuda(self.device)
                test_y = test_y.cuda(self.device)

            total += len(test_y)
            y_hat = test_net(test_x)
            test_pred = F.softmax(y_hat, dim=1).argmax(dim=1)  # size = (75)
            correct = torch.eq(test_pred, test_y).sum().item()
            correct_all += correct

        acc = correct_all / total

        print("acc:  "+str(acc))
        return res_source_trigger_target,res_source_trigger_source,res_none_trigger,acc


