# -*- coding: utf-8 -*-
'''
Created on 2018年10月30日

@author: zwp12
'''


import numpy as np;
import time;
from tools import SysCheck;
from new_ncf.ncf_param import NcfTraParmUST,NcfCreParamUST;
from new_ncf.nncf import simple_ncfUST,ncf_rnnUST;



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
    
    cp = NcfCreParamUST();
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
 
    time_range=(0,20);
 
    # 切分时间片
    rge=np.arange(time_range[0],time_range[1]);
    idx = np.where(np.isin(train_data[:,2],rge))[0];
    _train_data = train_data[idx]
    idx = np.where(np.isin(test_data[:,2],rge))[0];
    _test_data = test_data[idx];

    tp = NcfTraParmUST();
    tp.train_data=_train_data;
    tp.test_data=_test_data;

    tp.epoch=4;
    tp.batch_size=20;
    tp.learn_rate=0.02;
    tp.lr_decy_rate=1.0
    tp.lr_decy_step=int(n/tp.batch_size);
    tp.cache_rec_path=cache_path;
    tp.result_file_path=result_file;
    tp.load_cache_rec=True;
    tp.summary_path='summary'
    
    tp.rnn_unit=64;# rnn中隐特征的数量
    tp.seq_len=3;# 单时间序列长度
    tp.time_range=time_range;# 历史数据长度（开始，结束）
    tp.rnn_learn_rat=0.0007;# rnn的学习率
    tp.rnn_epoch=50;    # rnn的训练遍数


    rge=np.arange(time_range[1],time_range[1]+1);
    idx = np.where(np.isin(train_data[:,2],rge))[0];
    _train_data = train_data[idx]
    idx = np.where(np.isin(test_data[:,2],rge))[0];
    _test_data = test_data[idx];
    tp.ts_train_data=_train_data;
    tp.ts_test_data=_test_data;
  
    
    model = ncf_rnnUST(cp);
    
#     model.train_ncf(tp);
    model.conbine_train(tp);
    pass;

if __name__ == '__main__':
    run();
    pass

