# -*- coding: utf-8 -*-
'''
Created on 2018年10月24日

@author: zwp12
'''

import numpy as np;
import time;
from tools import SysCheck;
from tools import fwrite;

base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'

result='E:/work/Dataset/wst64/pmf.txt'

ust_shape= [142,4500,64];



def pmf(train,hid,eponch,lr,lamda):
    uNum,sNum=train.shape;
    P = np.random.normal(0,0.5,(uNum,hid))/np.sqrt(hid);
    Q = np.random.normal(0,0.5,(sNum,hid))/np.sqrt(hid);
    
    print('|-->训练开始，learn_rate=%f,repeat=%d'%(lr,eponch));
    
    
    U,S = np.where(train>0);
    datasize = len(U);
    
    for ep in range(eponch):
        tnow=time.time();
        losses=0.0;
        for i in range(datasize):
            u = U[i];
            s= S[i];
            rt = train[u,s];
            pt= np.sum(P[u]*Q[s]);
            eui = rt-pt;
            losses+=eui**2;
            #更改MF 参数
            tmp = lr * (eui*Q[s]-lamda*P[u]);
            Q[s] += lr * (eui*P[u]-lamda*Q[s]);
            P[u]+=tmp;
        losses/=datasize;
        print('|---->step%d 耗时%.2f秒 losses=%.6f|'%(ep+1,(time.time()-tnow),losses));
    
    return np.matmul(P,Q.T);
    
    


def mae(test,predict):
    testidx = np.where(test>0);
    y = test[testidx];
    py = predict[testidx];
    return np.sum(np.abs(py-y))/len(y);

def evel(test,predict):
    ret=[];
    allmae = mae(test,predict);
    ret.append(allmae);
    for i in range(test.shape[2]):
        tt = test[:,:,i];
        tp = predict[:,:,i];
        ret.append(mae(tt,tp));
    return ret;

spas=[15,20];
case=1;
eponch=1;

def run(spa,case,rid):

    train_path = base_path+'/Dataset/wst64/train_n/sparseness%.1f/training%d.txt'%(spa,case);
    test_path = base_path+'/Dataset/wst64/test_n/sparseness%.1f/test%d.txt'%(spa,case);

    print('加载训练集')
    train_data= np.loadtxt(train_path);
    print('加载测试集')
    test_data=np.loadtxt(test_path);
    n=len(train_data);
    
    print('转化训练集')
    train =  np.zeros(ust_shape);
    u = train_data[:,0].astype(int);
    s = train_data[:,1].astype(int);
    t = train_data[:,2].astype(int);
    train[u,s,t]=train_data[:,3];
    
    print('转化测试集')
    test =  np.zeros(ust_shape);
    u = test_data[:,0].astype(int);
    s = test_data[:,1].astype(int);
    t = test_data[:,2].astype(int);
    test[u,s,t]=test_data[:,3];    
    

    for rid in range(eponch):
        allret = {};
        print('开始 round-%d,spa-%.1f'%(rid,spa));
         
        print('预测开始');
        now = time.time();
        predict = np.zeros_like(train);
        for t in range(64):
            print('时间片%d预测开始'%t);
            predict[:,:,t] = pmf(train[:,:,t],32,50,0.001,0.01);
            print(mae(test[:,:,t], predict[:,:,t]));
        print('预测结束，耗时 %.2f秒\n'%((time.time() - now)))
         
        print('评测开始');
        now = time.time();
        ret = evel(test,predict);
        allret[spa] = ret;
        print('评测结束，耗时 %.2f秒\n'%((time.time() - now)))
        
        print(ret);

    for spa in allret:        
        retstr = '==================================\n';
        retstr+= 'round-%d spa-%.2f\n'%(rid,spa);
        rec = allret[spa];
        retstr+= 'all - mae \t%f\n'%(rec[0]);
        for i in range(1,len(rec)):
            retstr+= 't-%d mae \t%f\n'%(i,rec[i]);
        retstr+='localtime \t%s\n'%(time.asctime());    
        retstr+= '==================================\n\n';
        print(retstr);    
        fwrite.fwrite_append(result, retstr);
    
    pass;

if __name__ == '__main__':
    for rid in range(eponch):
        for spa in spas:
            run(spa,case,rid);
    pass