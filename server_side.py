import math
from scipy import spatial
import torch
from torchvision import models
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from PIL import Image
import torch.nn as nn
import pandas as pd
import os
import heapq
from sklearn.cluster import KMeans
from copy import deepcopy
from client_side import Client, model_name, dataset_name,NUM_LAYER,REQUIRE_JUDGE_LAYER
from collections import Counter
from models.vision import LeNet, ResNet, ResNet18, weights_init, ConvNet, LeNet_TS
import random
import torchvision
from torch.utils.data import DataLoader, Dataset
import random
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

#服务器端实现恶意客户端的筛选，并聚合生成全局模型
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


class Server(nn.Module):
    def __init__(self, device,malicious_rate, local_training_epoch=3, lr=0.001):

        super(Server, self).__init__()
        self.alpha = 0.5
        self.res_source_trigger_target=[]
        self.res_source_trigger_source=[]
        self.res_none_trigger=[]
        self.none_trigger_acc=[]
        self.device = device
        self.local_training_epoch = local_training_epoch
        self.malicious_rate=malicious_rate
        if dataset_name == 'mnist':
            self.net = LeNet().to(self.device)
        elif dataset_name == 'trafficsign':
            self.net = LeNet_TS().to(self.device)
        elif dataset_name=='CTSDB':
            self.net = LeNet_TS().to(self.device)
        self.loss_function = torch.nn.CrossEntropyLoss()

        self.clients = []
        self.RC_list = []
        self.mode_1 = "fed_train"
        self.mode_2 = "fed_test"
        self.batch_size = 20
        self.stimulus_each_class_num = 1
        self.path_now = os.path.dirname(__file__)
        self.last_path = '/final_test'  # -----------------------------------------------------
        if dataset_name == 'trafficsign':
            train_path = r'D:\cgx\security_test\trafficsign_train'
            self.test_data = torch.load(r'D:\cgx\security_test\trafficsign_test/1.pt')
            np.random.shuffle(self.test_data)


        elif dataset_name == 'mnist':
            train_path = r'D:\cgx\security_test\MNIST_train'
            self.test_data = torch.load(r'D:\cgx\security_test\MNIST_test/1.pt')

        elif dataset_name == 'CTSDB':
            train_path = r'D:\cgx\CTSDB\train_data'
            self.test_data = torch.load(r'D:\cgx\CTSDB\test_data/final_preprocess_test_data2.pt')

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


        self.time_accum = [0]

        if dataset_name == 'mnist':
            model = LeNet().to(self.device)
        elif dataset_name == 'trafficsign':
            model = LeNet_TS().to(self.device)
            #model=torch.load(r'D:\cgx\backdoor_resist\FedAVG\res_save\malicious0_alpha0_trafficsign/trafficsign_model_whole.pt')
        elif dataset_name=='CTSDB':
            model = LeNet_TS().to(self.device)

        self.stimulus_x, self.stimulus_y = self.prepare_stimulus_LFA(self.stimulus_each_class_num)

        self.num_clients = 10

        if (dataset_name=='mnist') or (dataset_name=='trafficsign'):
            train_file_set = os.listdir(train_path)
            train_path_set = [os.path.join(train_path, i) for i in train_file_set]
        elif dataset_name=='3D':
            train_path_set=[r'D:\cgx\3D_data\3D_train/train_data.pt' for i in range(self.num_clients)]
        elif dataset_name=='CTSDB':
            train_path_set=[r'D:\cgx\CTSDB\train_data/final_preprocess_train_data2.pt' for i in range(self.num_clients)]





        if self.alpha != 0:
            self.class_num_all = np.load(
                r'D:\cgx\security_test\dataset\diri_distribution/alpha' + str(self.alpha) + '.npy')
        else:
            self.class_num_all = []
            for i in range(self.num_clients):
                self.class_num_all.append([500 for i in range(10)])

        self.malicious_client_num = int(self.num_clients * self.malicious_rate)
        # self.malicious_client_id=random.sample([i for i in range(self.num_clients)],self.malicious_client_num)
        self.malicious_client_id = [i for i in range(self.malicious_client_num)]
        print("malicious client: ")
        print(self.malicious_client_id)
        for i in range(self.num_clients):
            if self.malicious_client_id.__contains__(i):
                self.clients.append(
                    Client(model=model, id=i, train_path=train_path_set[i], class_num_list=self.class_num_all[i],
                           learning_rate=lr, device=self.device, mode=self.mode_1,
                           batch_size=self.batch_size, client_type='malicious'))
            else:
                self.clients.append(
                    Client(model=model, id=i, train_path=train_path_set[i], class_num_list=self.class_num_all[i],
                           learning_rate=lr, device=self.device, mode=self.mode_1,
                           batch_size=self.batch_size, client_type='benign'))
        print(1)

    def server_add_trigger(self,image_matrix):
        row_start, row_end = 27, 30  # 行的开始和结束索引（包括在内）
        col_start, col_end = 27  , 30# 列的开始和结束索引（包括在内）
        for channel in range(3):
            image_matrix[channel][row_start:row_end + 1, col_start:col_end + 1] = 0
        return image_matrix

    def prepare_stimulus_LFA(self, each_num):
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

        # stimulus_list=[]
        # stimulus_list.append(self.source)
        # stimulus_list.append(self.target)
        '''
        classes_all.remove(self.source)
        classes_all.remove(self.target)
        stimulus_list=random.sample(classes_all,2)
        stimulus_list.append(self.source)
        stimulus_list.append(self.target)
        '''
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


        #stimulus_x=self.dataset_add_noise(stimulus_x,std=0.05)
        #stimulus_x=self.random_pixel_fill(stimulus_x,100)

       # stimulus_test=stimulus_x[0]
       # stimulus_test=stimulus_test.transpose(0,2)
        #stimulus_test1=stimulus_test.numpy()
        #plt.imshow(stimulus_x[1].transpose(0,2))
        #plt.show()
        print(1)


        if torch.cuda.is_available():
            stimulus_x = stimulus_x.cuda(self.device)
            stimulus_y = stimulus_y.cuda(self.device)


        return stimulus_x, stimulus_y

    def save_time(self, save_path):
        dataframe = pd.DataFrame(list(self.time_accum), columns=['time_accum'])
        dataframe.to_excel(save_path, index=False)


    #刺激數據集修改
    def img_add_gaussian_noise(self,img,std):

        for i in range(3):
            temp_gaussian_noise = torch.normal(0, std, img[i].shape)
            img[i]=img[i]+temp_gaussian_noise
        return img

    def dataset_add_noise(self,dataset,std):
        for i in range(len(dataset)):
            dataset[i]=self.img_add_gaussian_noise(dataset[i],std)
        return dataset

    def random_pixel_fill(self,dataset,amount):
        pixel_list=[]
        i=0
        while i<=amount:
            temp_pixel=[np.random.randint(0,32),np.random.randint(0,32),]
            if temp_pixel in pixel_list:
                continue
            else:
                pixel_list.append(temp_pixel)
                i=i+1

        for img in dataset:
            img=self.fill_pixel(img,pixel_list)
        return dataset

    def fill_pixel(self,img,pixel_list):
        for i in range(3):
            for pixel in pixel_list:
                img[i][pixel[0]][pixel[1]]=0
        return img


    def calculate_RC(self, model1, model2, stimulus_num):

        temp_model1 = deepcopy(model1)
        temp_model2 = deepcopy(model2)
        Layer_fea = {'layer3': 'feat3'}
        model1_Layer_Getter = torchvision.models._utils.IntermediateLayerGetter(temp_model1, Layer_fea)
        model2_Layer_Getter = torchvision.models._utils.IntermediateLayerGetter(temp_model2, Layer_fea)
        model1_out = temp_model1(self.stimulus_x)
        model2_out = temp_model2(self.stimulus_x)
        model1_softmax = torch.tensor(F.softmax(model1_out, dim=1), dtype=torch.float32)
        model2_softmax = torch.tensor(F.softmax(model2_out, dim=1), dtype=torch.float32)
        model1_RDM = torch.zeros((stimulus_num, stimulus_num)).to(self.device)
        model2_RDM = torch.zeros((stimulus_num, stimulus_num)).to(self.device)

        for i in range(stimulus_num):
            for j in range(stimulus_num):
                model1_RDM[i][j] = torch.cosine_similarity(model1_softmax[i].view(1, -1), model1_softmax[j].view(1, -1))
                model2_RDM[i][j] = torch.cosine_similarity(model2_softmax[i].view(1, -1), model2_softmax[j].view(1, -1))
        corr_result = self.corr2(model1_RDM, model2_RDM)
        if corr_result==1:
            print("warning2")
            print(1)
        return corr_result

        # 计算上传概率

    def mean2(self, x):
        y = torch.sum(x) / len(x)
        return y

    def corr2(self, a, b):
        a = a - self.mean2(a)
        b = b - self.mean2(b)
        r = torch.sum(a * b) / torch.sqrt(torch.sum(a * a) * torch.sum(b * b))
        return r

    def global_training(self, round):

        temp_RC = []
        id_train_0 = list(range(len(self.clients)))
        id_train = random.sample(id_train_0, int(len(id_train_0) * 1))  # clients of this round
        self.target=9

        source_list=random.sample([i for i in range(8)],5)

        #客户端完成本地训练
        for id, j in enumerate(id_train):
            self.clients[j].refresh(self.net)
            self.clients[j].local_fed_train(source_list,self.target)
            self.clients[j].epoch = round

        weight = []
        model_list = []
        vote_list = []
        print("\n轮次"+str(round)+"本地训练完成")
#*******************************客户端生成本地安全模型并上传*****************************************************************************************
        print("客户端开始生成本地安全模型")
        for id in id_train:
            # 使用stimulus刺激全局模型与本地模型
            self.clients[id].local_stimulate()
            self.clients[id].global_stimulate()
            # 计算RC
            self.clients[id].calculate_RC(round)

        upgrade_bool_dataframe = pd.DataFrame()#暂用

        filled_model_list=[]
       # for id in id_train:
        #    filled_model_list.append([deepcopy(self.clients[id].net),id])
        #使用全局模型层填充本地模型缺失的模型层

        for id in id_train:
            filled_local_model=deepcopy(self.clients[id].net)
            #判断是哪一模型层缺失
            for index,flag in enumerate(self.clients[id].upgrade_bool):
                if flag==0:
                    break
            #使用全局模型层进行填充
            for global_w, local_w in zip(self.net.named_parameters(), filled_local_model.named_parameters()):
                global_name, global_param = global_w
                local_name, local_param = local_w
                if global_name.__contains__(str(index + 1)):
                    local_param.data=torch.clone(global_param.data)
            filled_model_list.append([filled_local_model,id])


#*******************************服务器端接收到本地客户端上传的模型，进行恶意模型筛选*****************************************************************************************
        print("服务器端开始筛选恶意模型")

        RCV = np.zeros( [1,len(filled_model_list)])
        id_benign=[]
        # 计算RC矩阵
        for i in range(len(filled_model_list)):
            #print(filled_model_list[i][0]== self.net)
            RCV[0][i] = self.calculate_RC(filled_model_list[i][0], self.net,self.stimulus_each_class_num * 10)

        RCV_scaled = (RCV - RCV.min()) / (RCV.max() - RCV.min())
        print(RCV_scaled)
        print("轮次" + str(round) + "RCV计算完成")
        if np.isnan(RCV_scaled).any():
            id_benign=id_train
        else:

            #聚类分析筛选出恶意客户端
            cluster=KMeans(n_clusters=2)
            cluster_result=cluster.fit(RCV_scaled.reshape(-1,1)).labels_
            cluster_benign=Counter(cluster_result).most_common(1)[0][0]
            '''
            #選擇RC均值高的類別作爲正常類
            cluster_1_id=[]
            cluster_2_id=[]
            cluster_1_RC=[]
            cluster_2_RC=[]
            for i in range(len(filled_model_list)):
                if cluster_result[i]==0:
                    cluster_1_id.append(filled_model_list[i][1])
                    cluster_1_RC.append(RCV_scaled[0][i])
                else:
                    cluster_2_id.append(filled_model_list[i][1])
                    cluster_2_RC.append(RCV_scaled[0][i])
            id_benign=[]

            if (sum(cluster_1_RC)/len(cluster_1_RC))>(sum(cluster_2_RC)/len(cluster_2_RC)):
                id_benign=deepcopy(cluster_1_id)
            else:
                id_benign=deepcopy(cluster_2_id)

            '''

            #選擇客戶端數量多的為正常類
            id_benign=[]
            for i in range(len(filled_model_list)):
                if cluster_result[i]==cluster_benign:
                    id_benign.append(filled_model_list[i][1])

            print("Benign Client:  ")
            print(id_benign)
            if len(id_benign) == 0:
                id_benign = id_train

        for id, j in enumerate(id_benign):
            # weight.append(self.clients[j].size / size_all)
            weight.append(1 / len(id_benign))

        weight = np.array(weight)

        # *************************聚合生成全局模型************************************************************************************
        aggregate_list=[]
        for i in range(len(filled_model_list)):
            if filled_model_list[i][1] in id_benign:
                aggregate_list.append([filled_model_list[i][0],filled_model_list[i][1]])


        for i in range(len(aggregate_list)):
            for global_param, local_param in zip(self.net.parameters(), aggregate_list[i][0].parameters()):
                if (global_param is None or i == 0):
                    param_tem = Variable(torch.zeros_like(global_param)).to(self.device)
                    global_param.data.copy_(param_tem.data)
                if local_param is None:
                    local_param = Variable(torch.zeros_like(global_param)).to(self.device)
                global_param.data.add_(local_param.data * weight[i])
        print("聚合后全局模型测试结果：")
        res_source_trigger_target,res_source_trigger_source,res_none_trigger,acc = self.meta_test(source_list,self.target)
        self.res_source_trigger_target.append(res_source_trigger_target)
        self.res_source_trigger_source.append(res_source_trigger_source)
        self.res_none_trigger.append(res_none_trigger)
        self.none_trigger_acc.append(acc)
        print("Source Classified as Target:   " + str(res_source_trigger_target))
        print("Source Classification Accuracy:   " + str(res_source_trigger_source))
        print("None Source Classification Accuracy:   " + str(res_none_trigger))

    def meta_test(self, source_list,target):
        test_net = deepcopy(self.net)
        correct_all_trigger, total_trigger = 0.0, 0.0
        loss_all, correct_all, total = 0.0, 0.0, 0.0
        res_source_trigger_target, res_source_trigger_source, res_none_trigger = 0, 0, 0
        source_trigger_target, source_trigger_target_all = 0, 0
        source_trigger_source, source_trigger_source_all = 0, 0
        none_trigger, none_trigger_all = 0, 0

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
            print("With Trigger Overall Accuracy:  " + str(correct_all_trigger / total_trigger))


            for j in range(len(test_y)):
                if test_y[j] in source_list:
                    source_trigger_target_all = source_trigger_target_all + 1
                    source_trigger_source_all = source_trigger_source_all + 1
                    if test_pred[j] == target:
                        source_trigger_target = source_trigger_target + 1
                    elif test_pred[j] == test_y[j]:
                        source_trigger_source = source_trigger_source + 1
                else:
                    none_trigger_all = none_trigger_all + 1
                    if test_pred[j] == test_y[j]:
                        none_trigger = none_trigger + 1

        if self.malicious_rate == 0:
            res_none_trigger = none_trigger / none_trigger_all
        else:
            res_source_trigger_target = source_trigger_target / source_trigger_target_all
            res_source_trigger_source = source_trigger_source / source_trigger_source_all
            res_none_trigger = none_trigger / none_trigger_all


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

        print("No Trigger Overall Accuracy:  " + str(acc))
        return res_source_trigger_target, res_source_trigger_source, res_none_trigger, acc

    def save_ASR_acc(self):
        result_last_path = self.path_now + "/DPRPPFL_pattern/b64_no_pretrain_" + str(
            self.malicious_rate) + "_alpha" + str(self.alpha) + "_" + dataset_name + "_target_class" + str(self.target)
        if not os.path.exists(result_last_path):
            os.mkdir(result_last_path)

        source_trigger_target_file_path = result_last_path + "/source_as_target"
        source_trigger_source_file_path = result_last_path + "/source_accuracy"
        none_trigger_file_path = result_last_path + "/none_source_accuracy"
        none_trigger_acc_file_path = result_last_path + "/none_trigger_accuracy"

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
        np.save(source_trigger_target_file_path + "/source_as_target.npy", self.res_source_trigger_target)
        np.save(source_trigger_source_file_path + "/source_accuracy.npy", self.res_source_trigger_source)
        np.save(none_trigger_file_path + "/none_source_accuracy.npy", self.res_none_trigger)
        np.save(none_trigger_acc_file_path + "/none_trigger_accuracy.npy", self.none_trigger_acc)



    def forward(self):
        pass

