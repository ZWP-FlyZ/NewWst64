# -*- coding: utf-8 -*-
'''
Created on 2018年8月5日

@author: zwp12
'''

import numpy as np;
import time;
from ncf.Ncf import simple_ncf,hyp_ncf;
from ncf import NcfParam;
from ncf.NcfParam import NcfDatasoure


spa=3;
case=1;
train_path = 'E:/work/Dataset/ws/train_n/sparseness%d/training%d.txt'%(spa,case);
test_path = 'E:/work/Dataset/ws/test_n/sparseness%d/test%d.txt'%(spa,case);
cache_path = 'value_cache/spa%d_case%d.ckpt'%(spa,case);

def get_ncf_data(path):
    trdata = np.loadtxt(path, dtype=float);
    
    n = np.alen(trdata);
    print('加载数据成功 ',n);
    Uv=[];Sv=[];Yv=[];
    now = time.time();
    i=0;
    for item in trdata:
        tmp=[0]*339;
        tmp[int(item[0])]=1;
        Uv.append(tmp);
        tmp=[0]*5825;
        tmp[int(item[1])]=1;
        Sv.append(tmp);
        Yv.append([item[2]]);
        i+=1;
        if i % 1000 == 0:
            print('i=%d time=%.2f'%(i,time.time()-now))
            now = time.time();
    return n,(Uv,Sv,Yv);

def run():
    cp = NcfParam.NcfCreateParam;
    tp = NcfParam.NcfTrainParm;
    cp.us_shape=(339,5825);
    cp.hid_feat=32;
    cp.hid_units=[64,32,16];
    cp.drop_p=0.00001
    cp.reg_p=0.001
    
    
    ds = NcfDatasoure(train_path,test_path)
    n = ds.train_data_size;
   
    tp.datasize=n;
    tp.datasource=ds;
    tp.batchsize=16;
    
    tp.learn_loops=100;
    tp.learn_rate=0.05;
    tp.lr_decy_rate=0.999
    tp.lr_decy_step=int(n/tp.batchsize);
    tp.cache_rec_path=cache_path;
    tp.load_cache_rec=False;
    
    
    model = hyp_ncf(cp);
    
    model.train(tp);
    n,data = get_ncf_data(test_path);
    mae,py = model.calculate(data[0], data[1],data[2], cache_path);
    print(mae);
    print(py);
    print(np.array(data[2]))
    pass;




if __name__ == '__main__':
    run();
    pass