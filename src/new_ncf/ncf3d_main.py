# -*- coding: utf-8 -*-
'''
Created on 2018年8月8日

@author: zwp12
'''


import numpy as np;
import time;
from tools import SysCheck;
from new_ncf.ncf_param import NcfTraParm3D,NcfCreParam3D;
from new_ncf.ncf import hyb_ncf3D;


spa=2.5;
case=1;
base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'
train_path = base_path+'/Dataset/wst64/train_n/sparseness%.1f/training%d.txt'%(spa,case);
test_path = base_path+'/Dataset/wst64/test_n/sparseness%.1f/test%d.txt'%(spa,case);
cache_path = 'value_cache/wt64_spa%d_case%d.ckpt'%(spa,case);
dbug_paht = 'E:/work/Dataset/wst64/rtdata1.txt';

def run():
    cp = NcfCreParam3D();
    tp = NcfTraParm3D();
    cp.ust_shape=(142,4500,64);
    cp.hid_feat=32;
    cp.hid_units=[64,32,16];
    cp.drop_p=0.00001
    cp.reg_p=0.01
    
    print('加载训练集')
    train_data= np.loadtxt(train_path);
    print('加载测试集')
    test_data=np.loadtxt(test_path);
    n=len(train_data);
 
    tp.train_data=train_data;
    tp.test_data=test_data;

    tp.epoch=30;
    tp.batch_size=5;
    tp.learn_rate=0.01;
    tp.lr_decy_rate=0.99
    tp.lr_decy_step=int(n/tp.batch_size);
    tp.cache_rec_path=cache_path;
    tp.load_cache_rec=False;
    
    model = hyb_ncf3D(cp);
    
    model.train(tp);
    
    pass;

if __name__ == '__main__':
    run();
    pass