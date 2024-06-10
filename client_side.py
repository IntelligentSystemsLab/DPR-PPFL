import torch
import numpy as np
from torch import nn
import torchvision
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

model_name='lenet'
dataset_name='trafficsign'
#malicious_rate=0.4
if model_name=='lenet':
    LAYER_FEA={'layer1': 'feat1','layer2': 'feat2', 'layer3': 'feat3'}
    NUM_LAYER=3
    REQUIRE_JUDGE_LAYER=['layer1','layer2','layer3']
    HOOK_RES=['feat1','feat2','feat3']
elif model_name=='resnet18':
    LAYER_FEA={'layer1': 'feat1','layer2': 'feat2', 'layer3': 'feat3', 'layer4': 'feat4'}
    NUM_LAYER=4
    REQUIRE_JUDGE_LAYER=['layer1','layer2','layer3','layer4']
    HOOK_RES=['feat1','feat2','feat3','feat4']
elif model_name=='convnet':
#ConvNet
    LAYER_FEA={'layer1': 'feat1','layer2': 'feat2', 'layer3': 'feat3', 'layer4': 'feat4', 'layer5': 'feat5', 'layer6': 'feat6', 'layer7': 'feat7', 'layer8': 'feat8', 'layer9': 'feat9'}
    NUM_LAYER=9
    REQUIRE_JUDGE_LAYER=['layer1','layer2','layer3','layer4','layer5','layer6','layer7','layer8','layer9']
    HOOK_RES=['feat1','feat2','feat3','feat4','feat5','feat6','feat7','feat8','feat9']

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
    def __init__(self, model, id, train_path, learning_rate, device, mode,
                 batch_size,class_num_list,client_type):
        super(Client, self).__init__()
        self.id = id
        self.training_epoch=3
        self.net = deepcopy(model)
        self.learning_rate=learning_rate
        self.batch_size = batch_size
        self.RDV = []
        self.last_round_RC = []
        self.never_selected = 1
        self.RC = []
        self.upgrade_bool = []
        self.stimulus_each_class_num=1
        self.optim = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        self.device = device
        self.loss_function = torch.nn.CrossEntropyLoss().to(self.device)
        self.class_num_list=class_num_list
        self.client_type=client_type

        self.train_data = self.get_train_data(train_path, self.class_num_list, 1, 0)
        np.random.shuffle(self.train_data)

        self.time = 0
        self.epoch = 0


        self.stimulus_x, self.stimulus_y = self.prepare_stimulus_local(self.stimulus_each_class_num)

    def add_trigger(self,image_matrix):
        row_start, row_end = 27, 30  # 行的开始和结束索引（包括在内）
        col_start, col_end = 27, 30# 列的开始和结束索引（包括在内）
        for channel in range(3):
            image_matrix[channel][row_start:row_end + 1, col_start:col_end + 1] = 0
        return image_matrix

    def prepare_stimulus_local(self, each_num):
        if dataset_name == 'mnist':
            stimulus_data_ori = torch.load(r'D:\cgx\security_test\MNIST_stimulus/1.pt')
        elif dataset_name == 'trafficsign':
            stimulus_data_ori = torch.load(r'D:\cgx\security_test\trafficsign_stimulus/1.pt')
        elif dataset_name == 'CTSDB':
            stimulus_data_ori = torch.load(r'D:\cgx\CTSDB/stimulus.pt')
        stimulus_x = []
        stimulus_y = []
        categorize_flag = []
        classes_all = [i for i in range(10)]
        stimulus_list = [i for i in range(10)]

        for single_class in stimulus_list:
            categorize_flag.append([0, single_class])

        for i in range(len(stimulus_data_ori)):
            for class_flag in categorize_flag:
                if (stimulus_data_ori[i][1] == class_flag[1]) & (class_flag[0] < each_num):
                    if len(stimulus_x) == 0:
                        stimulus_x = stimulus_data_ori[i][0]
                        stimulus_y.append(stimulus_data_ori[i][1])
                    else:
                        stimulus_x = torch.cat([stimulus_x, stimulus_data_ori[i][0]], 0)
                        stimulus_y.append(stimulus_data_ori[i][1])

                    class_flag[0] = class_flag[0] + 1

        if dataset_name == 'mnist':
            stimulus_x = torch.reshape(stimulus_x, (each_num * 10, 1, 28, 28))
        elif dataset_name == 'trafficsign':
            stimulus_x = torch.reshape(stimulus_x, (each_num * 10, 3, 32, 32))
        elif dataset_name == 'CTSDB':
            stimulus_x = torch.reshape(stimulus_x, (each_num * 10, 3, 32, 32))
        stimulus_y = torch.tensor(stimulus_y)

        if torch.cuda.is_available():
            stimulus_x = stimulus_x.cuda(self.device)
            stimulus_y = stimulus_y.cuda(self.device)
        return stimulus_x, stimulus_y





    def get_train_data(self, path, class_num_list, source, target):
        each_class_num = list(class_num_list)
        #for i in range(len(each_class_num)):
         #   if self.client_type == 'malicious':
          #      each_class_num[i] = 500


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


    def forward(self):
        pass

    def local_fed_train(self,source_list,target):
        for _ in range(self.training_epoch):
            self.global_net = deepcopy(self.net)
            self.local_fea_out = []
            self.global_fea_out = []
            i=0
            # net_tem = deepcopy(self.net)
            # meta_optim_tem = torch.optim.Adam(net_tem.parameters(), lr = self.base_lr)
            if self.client_type=='malicious':
                #复制训练集用于发动后门攻击
                temp_local_set=deepcopy(self.train_data)

                for index in range(len(temp_local_set)):
                    if temp_local_set[index][1] in source_list:
                        temp_local_set[index] = list(temp_local_set[index])
                        # 加入trigger
                        temp_local_set[index][0] = self.add_trigger(temp_local_set[index][0])
                        # 修改标签
                        temp_local_set[index][1] = target
                        temp_local_set[index] = tuple(temp_local_set[index])
                local_set_size = int(len(temp_local_set) * 1.0)
                local_set = DatasetSplit(temp_local_set[:local_set_size])
                local_data_loader = DataLoader(local_set, batch_size=64, shuffle=True, drop_last=True)

            else:
                local_set_size = int(len(self.train_data) * 1.0)
                local_set = DatasetSplit(self.train_data[:local_set_size])
                local_data_loader = DataLoader(local_set, batch_size=64, shuffle=True, drop_last=True)


            for local_data in local_data_loader:
                local_data_x, local_data_y = local_data

                if torch.cuda.is_available():
                    local_data_x = local_data_x.cuda(self.device)
                    local_data_y = local_data_y.cuda(self.device)
                self.optim.zero_grad()

                output = self.net(local_data_x)
                # output = torch.squeeze(output)
                loss = self.loss_function(output, local_data_y)

                loss.backward()
                self.optim.step()
                self.optim.zero_grad()
                i += 1






    def local_stimulate(self):

        temp_local_net = deepcopy(self.net)
        Layer_fea = LAYER_FEA
        local_Layer_Getter = torchvision.models._utils.IntermediateLayerGetter(temp_local_net, Layer_fea)
        self.local_stimulus_out = local_Layer_Getter(self.stimulus_x)

        # 对当前训练轮次的全局模型进行stimulate

    def global_stimulate(self):
        temp_global_net = deepcopy(self.global_net)
        Layer_fea = LAYER_FEA
        global_Layer_Getter = torchvision.models._utils.IntermediateLayerGetter(temp_global_net, Layer_fea)
        self.global_stimulus_out = global_Layer_Getter(self.stimulus_x)

        # 计算本地模型的RDV

    def calculate_local_RDV(self, E_list):
        RDV = []
        for fea in HOOK_RES:
            temp_RDV = []
            temp = self.local_stimulus_out[fea].cpu().detach().numpy()
            for E_index in E_list:
                temp_RDV.append(np.linalg.norm(temp[E_index[0]] - temp[E_index[1]]))
            RDV.append(temp_RDV)
        return RDV
        # 计算全局模型的RDV

    def calculate_global_RDV(self, E_list):
        RDV = []
        for fea in HOOK_RES:
            temp_RDV = []
            temp = self.global_stimulus_out[fea].cpu().detach().numpy()
            for E_index in E_list:
                temp_RDV.append(np.linalg.norm(temp[E_index[0]] - temp[E_index[1]]))
            RDV.append(temp_RDV)
        return RDV
        # 计算RC
    def generate_full_upgrade(self):
        self.upgrade_bool = [1 for i in range(NUM_LAYER)]
    def calculate_RC(self, round):
        # self.RC=random.random()
        self.RC = []
        E_list = self.generate_E(20)
        local_RDV = self.calculate_local_RDV(E_list)
        global_RDV = self.calculate_global_RDV(E_list)
        for i in range(NUM_LAYER):
            self.RC.append(np.square(np.corrcoef(global_RDV[i], local_RDV[i])[0, 1]))
        if self.never_selected == 1:
            self.upgrade_bool = [1 for i in range(NUM_LAYER)]
            self.upgrade_bool[np.random.randint(0, high=NUM_LAYER)] = 1
            self.last_round_RC = deepcopy(self.RC)
            self.never_selected = 0
        else:
            last = deepcopy(self.last_round_RC)
            this = deepcopy(self.RC)

            self.last_round_RC = deepcopy(self.RC)
            temp_delta = []
            for i in range(NUM_LAYER):
                temp_delta.append(np.abs((last[i] - this[i]) / this[i]))
            self.upgrade_bool = [1 for i in range(NUM_LAYER)]
            self.upgrade_bool[temp_delta.index(min(temp_delta))] = 0  #相似度最低的层不上传

        # 计算上传概率

    def calculate_pro(self):
        pass
        # 生成RDV元素的索引

    def generate_E(self, E_number):
        i = 0
        E_list = []
        while i < E_number:
            E_element = [random.randint(0, 9), random.randint(0, 9)]
            if E_list.__contains__(E_element):
                continue
            else:
                E_list.append(E_element)
                i = i + 1
        return E_list
        # 本地模型hook

    def local_hook(self, module, input, output):
        #     print("local hooking")
        self.local_fea_out.append(output)
        #      self.local_fea_in.append(input)
        # 全局模型hook

    def global_hook(self, module, input, output):
        #    print("global hooking")
        self.global_fea_out.append(output)
        #     self.global_fea_in.append(input)


    def refresh(self, model):
        for w, w_t in zip(self.net.parameters(), model.parameters()):
            w.data.copy_(w_t.data)










