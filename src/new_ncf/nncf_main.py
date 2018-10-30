# -*- coding: utf-8 -*-
'''
Created on 2018年10月30日

@author: zwp12
'''


import numpy as np;
import time;
from tools import SysCheck;
from new_ncf.ncf_param import NcfTraParmUST,NcfCreParamUST;
from new_ncf.nncf import simple_ncfUST;



spa=5.0;
case=1;
base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'
train_path = base_path+'/Dataset/wst64/train_n/sparseness%.1f/training%d.txt'%(spa,case);
test_path = base_path+'/Dataset/wst64/test_n/sparseness%.1f/test%d.txt'%(spa,case);
cache_path = 'value_cache/wt64_spa%d_case%d.ckpt'%(spa,case);
result_file= 'result/wst64_spa%.1f_case%d.txt'%(spa,case);
dbug_paht = 'E:/work/Dataset/wst64/rtdata1.txt';


def run():
    cp = NcfTraParmUST();
    tp = NcfCreParamUST();
    cp.ust_shape=(142,4500,64);
    cp.hid_feat=32;
    cp.hid_units=[64,32,16];
    cp.drop_p=0.00001
    cp.reg_p=0.00001
    
    print('加载训练集')
    train_data= np.loadtxt(train_path);
    print('加载测试集')
    test_data=np.loadtxt(test_path);
    n=len(train_data);
 
#     rge=np.arange(0,20);
#     idx = np.where(np.isin(train_data[:,2],rge))[0];
#     train_data = train_data[idx]
#     idx = np.where(np.isin(test_data[:,2],rge))[0];
#     test_data = test_data[idx];
    
    tp.train_data=train_data;
    tp.test_data=test_data;

    tp.epoch=30;
    tp.batch_size=20;
    tp.learn_rate=0.03;
    tp.lr_decy_rate=1.0
    tp.lr_decy_step=int(n/tp.batch_size);
    tp.cache_rec_path=cache_path;
    tp.result_file_path=result_file;
    tp.load_cache_rec=False;
    tp.summary_path='summary'
    
    model = simple_ncfUST(cp);
    
    model.train_ncf( tp);
    
    pass;

if __name__ == '__main__':
    run();
    pass

