# -*- coding: utf-8 -*-
'''
Created on 2018年9月12日

@author: zwp12
'''

import numpy as np;
import time;
from tools import SysCheck;

from new_ncf.ncf_param import NcfTraParm,NcfCreParam;
from new_ncf.ncf_more import ncf_pp;


spa=5;
case=1;
base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'

train_path = base_path+'/Dataset/ws/train_n/sparseness%d/training%d.txt'%(spa,case);
test_path = base_path+'/Dataset/ws/test_n/sparseness%d/test%d.txt'%(spa,case);
cache_path = 'value_cache/spa%d_case%d.ckpt'%(spa,case);
result_file= 'result/ws_spa%.1f_case%d.txt'%(spa,case);
dbug_paht = 'E:/work/Dataset/wst64/rtdata1.txt';

def run():
    cp = NcfCreParam();
    tp = NcfTraParm();
    cp.us_shape=(339,5825);
    cp.hid_feat=32;
    cp.hid_units=[64,32,16];
    cp.drop_p=0.00001
    cp.reg_p=0.0001
    cp.baseline_mu=0.0;
    

    train_data= np.loadtxt(train_path);
    test_data=np.loadtxt(test_path);
    
    cp.baseline_mu=np.mean(train_data[:,2]);
    
#     u_invked = [[] for _ in range(cp.us_shape[0])];
#     u_invked_cot=[];
#     for usv in train_data:
#         u_invked[int(usv[0])].append(int(usv[1]));
#     for uiv in u_invked:
#         u_invked_cot.append(len(uiv));
    
    R = np.zeros(cp.us_shape);
    u = train_data[:,0].astype(np.int32);
    s = train_data[:,1].astype(np.int32);
    R[u,s]=1.0;
    nonzeroes = np.count_nonzero(R, axis=1);
    noz = 1.0/np.sqrt(nonzeroes);
    noz = np.reshape(noz,[-1,1]);
    us_invked=(R*noz).astype(np.float32);


    
    n=len(train_data);

    tp.train_data=train_data;
    tp.test_data=test_data;
    tp.epoch=30;
    tp.batch_size=5;
    tp.learn_rate=0.01;
    tp.lr_decy_rate=1.0
    tp.lr_decy_step=int(n/tp.batch_size);
    tp.cache_rec_path=cache_path;
    tp.result_file_path= result_file;
    tp.load_cache_rec=False;
    tp.us_invked = us_invked;
#     tp.u_invked=u_invked;
#     tp.u_invked_cot=u_invked_cot;
    
    
    model = ncf_pp(cp);

    model.train(tp);
    
    pass;

if __name__ == '__main__':
    run();
    pass