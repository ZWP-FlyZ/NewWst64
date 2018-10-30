# -*- coding: utf-8 -*-
'''
Created on 2018年10月24日

@author: zwp12
'''

import numpy as np;
import time;
from tools import SysCheck;
from tools import fwrite;

epon=3;
spas=[5,10,15,20];
base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'
dbug_paht = 'E:/work/Dataset/wst64/rtdata.txt';

result='E:/work/Dataset/wst64/pmf.txt'

def load_origin_data(path):
    rec_table = np.loadtxt(path,float); 
    return rec_table
    pass;

def splitMat(mat,spa,seed):
    idxx,idxy = np.where(mat>0);
    recNum = len(idxx);
    train_size= int((spa/100.0) * recNum);
    test_size =recNum - train_size;
    chooseidx = np.arange(recNum);
    np.random.seed(seed);
    np.random.shuffle(chooseidx);
    train_idx = chooseidx[0:train_size];
    test_idx = chooseidx[-test_size:];
#     print('train_size=%d,test_size=%d'%(train_size,test_size))
    train_u = idxx[train_idx];
    train_s = idxy[train_idx];
    test_u = idxx[test_idx];
    test_s = idxy[test_idx];
    train= np.zeros_like(mat);
    train[train_u,train_s]=mat[train_u,train_s];
    test= np.zeros_like(mat);
    test[test_u,test_s]=mat[test_u,test_s];
    
    # 重要更改
    idxX = (np.sum(train, axis=1) == 0)
    test[idxX, :] = 0
    idxY = (np.sum(train, axis=0) == 0)
    test[:, idxY] = 0   

    return train,test;


def spliter(R,spa,rid):
    tNum = R.shape[2];
    trainTen =  np.zeros_like(R);
    testTen = np.zeros_like(R);
    for i in range(tNum):
        seed = rid+i*1000;
        rmat = R[:,:,i];
        trainMat,testMat = splitMat(rmat,spa,seed);
        trainTen[:,:,i]=trainMat;
        testTen[:,:,i]=testMat;

    return trainTen,testTen;



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

def run():
    
    print ('加载数据开始');
    now = time.time();
    rec_table = load_origin_data(dbug_paht);
    recNum = len(rec_table);
    print('原始数据：\n',rec_table);
    print('总记录数：%d'%(recNum))
    R = np.zeros([142,4500,64]);\
    u = rec_table[:,0].astype(int);
    s = rec_table[:,1].astype(int);
    t = rec_table[:,2].astype(int);
    R[u,s,t]=rec_table[:,3];
    print ('加载数据完成，耗时 %.2f秒\n'%((time.time() - now)));
    
    for rid in range(epon):
        allret = {};
        for spa in spas:
            print('开始 round-%d,spa-%.1f'%(rid,spa));
             
            print('分割数据集开始');
            now = time.time();
            train,test = spliter(R,spa,rid);
            print('分割数据集结束，耗时 %.2f秒\n'%((time.time() - now)))
             
            print('预测开始');
            now = time.time();
            predict = np.zeros_like(train);
            for t in range(64):
                print('时间片%d预测开始'%t);
                predict[:,:,t] = pmf(train[:,:,t],32,50,0.01,0.001);
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
    run();
#     a = np.array([1,1,0,0,2]);
#     b = np.array([0,2,1,0,3]);
#     
#     print(getpcc(a,b,0.8,1.2));


#     a = np.reshape(np.arange(900),[30,30]).astype(np.float)/15.0;
#     print(pmf(a,10,100,0.01,0.0001));




    pass