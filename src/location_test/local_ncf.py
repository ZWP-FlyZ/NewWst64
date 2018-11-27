# -*- coding: utf-8 -*-
'''
Created on 2018年9月5日

@author: zwp12
'''


import numpy as np;
import time;
import math;
import os;
from tools import SysCheck;

from new_ncf.ncf_param import NcfTraParm,NcfCreParam;
from new_ncf.ncf import hyb_ncf_local;

from new_ncf.ncf import hyb_ncf_local_;


from location import localtools
from new_ncf.ncf3d_main import spa


base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'
origin_data = base_path+'/rtdata.txt';

spas = [2];

def mf_base_run(spa,case):
    train_path = base_path+'/Dataset/ws/train_n/sparseness%.1f/training%d.txt'%(spa,case);
    test_path = base_path+'/Dataset/ws/test_n/sparseness%.1f/test%d.txt'%(spa,case);
    cache_path = 'value_cache/spa%d_case%d.ckpt'%(spa,case);
    result_file= 'result/ws_spa%.1f_case%d.txt'%(spa,case);
    dbug_paht = 'E:/work/Dataset/wst64/rtdata1.txt';
    
    loc_classes = base_path+'/Dataset/ws/localinfo/ws_classif_out.txt';
    
    print('开始实验，稀疏度=%.1f,case=%d'%(spa,case));
    print ('加载训练数据开始');
    now = time.time();
    trdata = np.loadtxt(train_path, dtype=float);
    ser_class = localtools.load_classif(loc_classes);
    classiy_size = len(ser_class);
    n = np.alen(trdata);
    print ('加载训练数据完成，耗时 %.2f秒，数据总条数%d  \n'%((time.time() - now),n));
    
    
    print ('加载测试数据开始');
    tnow = time.time();
    ttrdata = np.loadtxt(test_path, dtype=float);
    tn = np.alen(ttrdata);
    print ('加载测试数据完成，耗时 %.2f秒，数据总条数%d  \n'%((time.time() - tnow),tn));
    
    print ('分类数据集开始');
    tnow = time.time();
    train_sets = localtools.data_split_class(ser_class, trdata);
    test_sets = localtools.data_split_class(ser_class, ttrdata);
    del trdata,ttrdata;
    print ('分类数据集结束，耗时 %.2f秒  \n'%((time.time() - tnow)));
    
    
    cp = NcfCreParam();
    tp = NcfTraParm();
    cp.us_shape=(339,5825);
    cp.hid_feat=32;
    cp.hid_units=[64,32,16];
    cp.drop_p=0.00001
    cp.reg_p=0.0001
    
        
    tp.train_data=train_sets;
    tp.test_data=test_sets;
    tp.epoch=40;
    tp.batch_size=5;
    tp.learn_rate=0.015;
    tp.lr_decy_rate=1.0
    tp.lr_decy_step=int(n/tp.batch_size);
    tp.cache_rec_path=cache_path;
    tp.result_file_path= result_file;
    tp.load_cache_rec=False;
    tp.classif_size = len(train_sets);
    
    
    print ('训练模型开始');
    tnow = time.time();
    model = hyb_ncf_local_(cp);
    
    model.train(tp);
                     
    print ('训练模型结束，耗时 %.2f秒  \n'%((time.time() - tnow)));  

   

    print('实验结束，总耗时 %.2f秒,稀疏度=%.1f\n'%((time.time()-now),spa));


if __name__ == '__main__':
    for spa in spas:
        for ca in range(2,5):
            case = ca;
            mf_base_run(spa,case);
