# -*- coding: utf-8 -*-
'''
Created on 2018年8月7日

@author: zwp12
'''

import numpy as np;
import time;

from new_ncf.ncf_param import NcfTraParm,NcfCreParam;
from new_ncf.ncf import hyb_ncf;


spa=10;
case=1;
train_path = 'E:/work/Dataset/ws/train_n/sparseness%d/training%d.txt'%(spa,case);
test_path = 'E:/work/Dataset/ws/test_n/sparseness%d/test%d.txt'%(spa,case);
cache_path = 'value_cache/spa%d_case%d.ckpt'%(spa,case);
dbug_paht = 'E:/work/Dataset/wst64/rtdata1.txt';

def run():
    cp = NcfCreParam();
    tp = NcfTraParm();
    cp.us_shape=(339,5825);
    cp.hid_feat=32;
    cp.hid_units=[64,32,16];
    cp.drop_p=0.00001
    cp.reg_p=0.01
    

    train_data= np.loadtxt(dbug_paht);
    test_data=np.loadtxt(dbug_paht);
    n=len(train_data);

    tp.train_data=train_data;
    tp.test_data=test_data;
    tp.epoch=30;
    tp.batch_size=1;
    tp.learn_rate=0.01;
    tp.lr_decy_rate=0.96
    tp.lr_decy_step=int(n/tp.batch_size);
    tp.cache_rec_path=cache_path;
    tp.load_cache_rec=False;
    
    model = hyb_ncf(cp);
    
    model.train(tp);
    
    pass;

if __name__ == '__main__':
    run();
    pass