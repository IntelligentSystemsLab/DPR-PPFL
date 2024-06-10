import numpy as np
import torch
import pandas as pd
from FedAVG_server import Metanet, model_name, dataset_name
import time
import os
import warnings

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
start = time.time()


def main(active_rate,malicious_rate):
    epoch = 100
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    time_list = []
    meta_net = Metanet(device=device,active_rate=active_rate,malicious_rate=malicious_rate)
    for i in range(epoch):
        time_start = time.time()
        if (i + 1) % 20 == 0 or (i + 1) <= 5:
            print("{} round training.".format(i + 1))
        meta_net.meta_training(i + 1)
        print('running time:', time.time() - start, 's')

        time_list.append(time.time() - time_start)

    meta_net.save_ASR_acc()
    del meta_net


if __name__ == '__main__':
    rate=[0,0.1,0.2,0.3,0.4]
    for r in rate:

        main( active_rate=1,malicious_rate=r)

    '''
    thres_list=[0.7,0.75,0.8,0.85,0.9,0.95]
    for thres in thres_list:
        main(thres)
    end = time.time()
    run_time = end - start
    print('running time:',run_time,'s')
    '''
    # path_origin = r"D:\cgx\security_test\fedavg_attack\FedAVG\result_save_fedavg/0.4_mnist_noniid2_thres0.75"
    '''
    path_origin = r"D:\cgx\security_test\fedavg_attack\FedAVG\result_save_fedavg/heatmap_short/0.4_alpha0.5_"+dataset_name

    label_source=[i for i in range(10)]
    label_target = [i for i in range(10)]
    active_rate=1

    for source in label_source:
        for target in label_target:
            if source==target:
                continue
            #elif (source==9)&(target==7):
            #    continue
            path = path_origin + "_source_class" + str(source) + "_target_class" + str(target)
            if os.path.exists(path):
                if os.path.exists(path+ "/ASR/ASR.npy"):
                    temp = np.load(path + "/ASR/ASR.npy")

                   # if temp[-1]<0.1:
                    #    print("source"+str(source)+"    target"+str(target))
                     #   main(source_class=source,target_class=target,active_rate=1)
            else:
                print(str(source)+str(target))
                main(source_class=source, target_class=target,active_rate=1)

    active_rate=[1]
    for rate in active_rate:
        for source in label_source:
            for target in label_target:
                if source==target:
                    continue
                else:

                    main(source_class=source,target_class=target,active_rate=rate)
    '''
