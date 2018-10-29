# -*- coding: utf-8 -*-
'''
Created on 2018年8月4日

@author: zwp12
'''

'''
WSDream 64 时间数据集随机分割方案

'''


import time;
import numpy as np;
import random;
import os;
from tools import SysCheck;

base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'
    
origin_data_path = base_path+'/Dataset/wst64/rtdata.txt';
train_output_path = base_path+'/Dataset/wst64/train_n'
test_output_path = base_path+'/Dataset/wst64/test_n'

# 分割稀疏度列表
spa_list=[1.0,2.5,5.0,10.0];

# 每个稀疏度需要的例子数
case_cout=10;


# 加载数据集
def load_origin_data(path):
    rec_table = np.loadtxt(path,float); 
    return rec_table
    pass;

def save_data(path,data):
    np.savetxt(path,data,'%d %d %d %.3f');
    pass;


def splitMat(mat,spa):
    idxx,idxy = np.where(mat>0);
    recNum = len(idxx);
    train_size= int((spa/100.0) * recNum);
    test_size = train_size;
    chooseidx = np.arange(recNum);
    np.random.shuffle(chooseidx);
    train_idx = chooseidx[0:train_size];
    test_idx = chooseidx[recNum-test_size:recNum];
    print('train_size=%d,test_size=%d'%(train_size,test_size))
    train_u = idxx[train_idx];
    train_s = idxy[train_idx];
    test_u = idxx[test_idx];
    test_s = idxy[test_idx];
    train= np.zeros_like(mat);
    train[train_u,train_s]=mat[train_u,train_s];
    test= np.zeros_like(mat);
    test[test_u,test_s]=mat[test_u,test_s];
    return train,test;
    

def run():
    
    print('开始分割!分割序列=',spa_list);
    
    
    print ('加载数据开始');
    now = time.time();
    rec_table = load_origin_data(origin_data_path);
    recNum = len(rec_table);
    print('原始数据：\n',rec_table);
    print('总记录数：%d'%(recNum))
    R = np.zeros([142,4500,64]);\
    u = rec_table[:,0].astype(int);
    s = rec_table[:,1].astype(int);
    t = rec_table[:,2].astype(int);
    R[u,s,t]=rec_table[:,3];
    print ('加载数据完成，耗时 %.2f秒\n'%((time.time() - now)));
    
       
    for spa in spa_list:
        train_spa_path = train_output_path+'/sparseness%.1f'%(spa);
        test_spa_path = test_output_path+'/sparseness%.1f'%(spa);
        if not os.path.isdir(train_spa_path):
            os.makedirs(train_spa_path);
        if not os.path.isdir(test_spa_path):
            os.makedirs(test_spa_path); 
            
        for case in range(1,case_cout+1):
            nR = np.zeros_like(R);
            ntR = np.zeros_like(R);
            print('开始分割数据    sap=%.1f \t case=%d'%(spa,case));
            for t in range(64):
                np.random.seed();
                p = R[:,:,t];
                trainmat,testmat= splitMat(p,spa);
                nR[:,:,t] += trainmat;
                ntR[:,:,t] += testmat;
                

            train_file = train_spa_path+'/training%d.txt'%(case);
            test_file = test_spa_path+'/test%d.txt'%(case);
            u,s,t = np.where(nR>0);
            rt = nR[u,s,t];
            tmpdata = np.c_[u,s,t,rt];
            save_data(train_file,tmpdata);
            u,s,t = np.where(ntR>0);
            rt = ntR[u,s,t];
            tmpdata = np.c_[u,s,t,rt];
            save_data(test_file,tmpdata);
        print('\n');
    pass;

def run_():
    
    print('开始分割!分割序列=',spa_list);
    
    
    print ('加载数据开始');
    now = time.time();
    rec_table = load_origin_data(origin_data_path);
    recNum = len(rec_table);
    print('原始数据：\n',rec_table);
    print('总记录数：%d'%(recNum))
    print ('加载数据完成，耗时 %.2f秒\n'%((time.time() - now)));
    
    
    
    
    for spa in spa_list:
        train_spa_path = train_output_path+'/sparseness%.1f'%(spa);
        test_spa_path = test_output_path+'/sparseness%.1f'%(spa);
        if not os.path.isdir(train_spa_path):
            os.makedirs(train_spa_path);
        if not os.path.isdir(test_spa_path):
            os.makedirs(test_spa_path); 
            
        train_data_size = int((spa/100.0) * recNum);    
        test_data_size = train_data_size;
        print('train_size=%d,test_size=%d'%(train_data_size,test_data_size))     
        for case in range(1,case_cout+1):
            allrec = np.arange(0,recNum,dtype=np.int);
            print('开始分割数据    sap=%.1f \t case=%d'%(spa,case));
            np.random.shuffle(allrec);
            train_file = train_spa_path+'/training%d.txt'%(case);
            test_file = test_spa_path+'/test%d.txt'%(case);
            tmpdata = rec_table[allrec[0:train_data_size]];
            save_data(train_file,tmpdata);
            tmpdata = rec_table[allrec[recNum-test_data_size:recNum]];
            save_data(test_file,tmpdata);
        print('\n');
    pass;



if __name__ == '__main__':
    run();

#     a = np.arange(30).reshape([5,6]);
#     print(a);
#     print(splitMat(a,30))


    pass