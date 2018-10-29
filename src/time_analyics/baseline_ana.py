# -*- coding: utf-8 -*-
'''
Created on 2018年10月22日

@author: zwp12
'''

'''
观察基准线分析
'''

import numpy as np;
import time;
from tools import SysCheck;

spa=10.0;
case=1;
base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'
train_path = base_path+'/Dataset/wst64/train_n/sparseness%.1f/training%d.txt'%(spa,case);
test_path = base_path+'/Dataset/wst64/test_n/sparseness%.1f/test%d.txt'%(spa,case);
cache_path = 'value_cache/wt64_spa%d_case%d.ckpt'%(spa,case);
result_file= 'result/wst64_spa%.1f_case%d.txt'%(spa,case);
dbug_paht = 'E:/work/Dataset/wst64/rtdata1.txt';


ust_shape= [142,4500,64];

def run():
    
    print('加载训练集')
    train_data= np.loadtxt(train_path);
    print('加载测试集')
    test_data=np.loadtxt(test_path);
    n=len(train_data);
    
    print('转化训练集')
    train_mat =  np.zeros(ust_shape);
    u = train_data[:,0].astype(int);
    s = train_data[:,1].astype(int);
    t = train_data[:,2].astype(int);
    train_mat[u,s,t]=train_data[:,3];
    
    print('转化测试集')
    test_mat =  np.zeros(ust_shape);
    u = test_data[:,0].astype(int);
    s = test_data[:,1].astype(int);
    t = test_data[:,2].astype(int);
    test_mat[u,s,t]=test_data[:,3];    
    
#     print(train_mat);
#     print(test_mat);
    
    # 测试各个时间片下，基准测试
#     predict = train_mat.copy();
#     for t in range(ust_shape[2]):
#         p = train_mat[:,:,t];
#         sum = np.sum(p,axis=0);
#         cot= np.count_nonzero(p,axis=0);
#         mean = np.divide(sum,cot,out=np.zeros_like(sum),
#                         where = cot != 0);
#         print(mean);
#         for s in range(ust_shape[1]):
#             idx = np.where(predict[:,s,t]<=0);
#             predict[idx[0],s,t] = mean[s];
#     print(predict);
    
    # 每个服务下，计算过所有用户与时间片的平均值填补
    predict = train_mat.copy();
    for s in range(ust_shape[1]):
        p = train_mat[:,s,:];
        sum = np.sum(p);
        cot= np.count_nonzero(p);
        mean = sum/(np.sum(p>0)++ np.spacing(1));
        predict[:,s,:]=mean;


    # 每个服务下，计算过所有用户与时间片的平均值填补
#     predict = train_mat.copy();
#     for s in range(ust_shape[0]):
#         p = train_mat[s,:,:];
#         sum = np.sum(p);
#         cot= np.count_nonzero(p);
#         mean = sum/cot;
#         predict[s,:,:]=mean;
       
       
    # 每个服务下，计算过所有用户与时间片的平均值填补
#     predict = train_mat.copy();
#     for s in range(ust_shape[2]):
#         p = train_mat[:,:,s];
#         sum = np.sum(p);
#         cot= np.count_nonzero(p);
#         mean = sum/cot;
#         predict[:,:,s]=mean;

 
    
    sli64=[];
    for t in range(ust_shape[2]):
        y = test_mat[:,:,t];
        py = predict[:,:,t];
        whr = y>0;
        delt = np.subtract(py,y,out=np.zeros_like(py),where=whr);
        delt = np.abs(delt);
        mae = np.sum(delt)/np.sum(y>0);
        sli64.append(mae);
    
    allmae = [];
    y = test_mat;
    py = predict;
    idx = np.where(y);
    y = test_mat[idx];
    py = predict[idx];
    delt = np.abs(y-py);
    mae = np.sum(delt)/len(delt);
    allmae.append(mae);
    print(allmae)
    
    pass;
    




if __name__ == '__main__':
    
    run();
    pass