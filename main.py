import numpy as np
import torch
import pandas as pd
from server_side import Server
import time
import os
import warnings
warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
start = time.time()

def main(malicious_rate):
    epoch =300
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu' )
    global_server = Server(device=device,malicious_rate=malicious_rate)
    for i in range(epoch):
        time_start=time.time()

        if (i+1)%20 == 0 or (i+1)<=5:
            print("{} round training.".format(i+1))
        global_server.global_training(i+1)
        print('running time:', time.time()-start, 's')


    global_server.save_ASR_acc()
#    meta_net.temp_save_res()
    del global_server

 #   meta_net.save_res()
if __name__ == '__main__':
    #thres_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    #for thres in thres_list:
    rate=[0,0.1,0.2,0.3,0.4]
    for r in rate:
        main(malicious_rate=r)
    #main(source_class=5, target_class=4)
    '''
    label_source=[i for i in range(10)]
    label_target = [i for i in range(10)]
    for source in label_source:
        for target in label_target:
            if source==target:
                continue
            elif os.path.exists(r'D:\cgx\security_test\fedavg_attack\RSFL_short\result_save_RSFL_short/0.4_alpha0.5_trafficsign_source_class'+str(source)+'_target_class'+str(target)):
                continue
            else:
                print(str(source))
                print(str(target))
                main(source_class=source, target_class=target)
    '''

