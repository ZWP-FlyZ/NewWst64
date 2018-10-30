# -*- coding: utf-8 -*-
'''
Created on 2018年10月23日

@author: zwp12
'''

'''
UPCC测试
'''
import numpy as np;
import time;
from tools import SysCheck;
from tools import fwrite;

epon=2;
spas=[5.0,10,15,20.0];
base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'
dbug_paht = 'E:/work/Dataset/wst64/rtdata.txt';

result='E:/work/Dataset/wst64/ipcc.txt'

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

def baseline_predict(train):
    predict = np.zeros_like(train);
    sNum = train.shape[1];
    for i in range(sNum):
        rmat = train[:,i,:];
        idx = np.where(rmat<=0);
        mean = np.sum(rmat)/(np.sum(rmat>0)+np.spacing(1));
        predict[idx[0],i,idx[1]]=mean;
    
    return predict;


def getpcc(ui,uj,imean,jmean):
    
    # 计算共同调用元素，若共同元素小于2，返回0
    
    tmp = np.logical_and(ui>0,uj >0);
    idx = np.where(tmp)[0];  
    if len(idx)<2:return 0;
    
    nui = ui[idx]-imean;
    nuj = uj[idx]-jmean;
    
    uppall = np.sum(nui*nuj);
    downalla = np.sum(nui*nui);
    downallb = np.sum(nuj*nuj);
    downall = np.sqrt(downalla * downallb);
    if downall==0:return 0;
    return uppall / downall;

def upcc(train,topK):
    uNum,sNum = train.shape;
    
    predict = np.zeros_like(train);
    uMean = np.sum(train,axis=1)/(np.sum(train>0,axis=1)+np.spacing(1));
    
    W = np.zeros([uNum,uNum]);
    for i in range(uNum):
        ui_rec=[];
#         if i %20==0:
#             print('开始用户%d的相似用户计算'%i);
        for j in range(uNum):
            # 不考虑无任何调用记录的用户
            if uMean[i]==0 or uMean[j]==0:continue;
            pccv= 0.0;
            if j>i:
                pccv = getpcc(train[i],train[j],uMean[i],uMean[j]);
                W[i,j]=pccv;
                W[j,i]=pccv;
            elif j<i:
                pccv = W[i,j];
            
            # 不考虑小于0的
            if pccv>0:
                ui_rec.append([j,pccv]);
            
#         if i %20==0:
#             print('开始用户%d的预测'%i);
        soreduser = sorted(ui_rec,key=lambda p:p[1],reverse=True);
        for s in range(sNum):
            if train[i][s] >0:continue;# 忽略已有数据预测
            k=0;
            pccsum=0;
            prev=0;
            for item in soreduser:
                uid,pcc = item;
                if k>=topK:break;# 多出元素
                if train[uid][s] <=0:continue; # 忽略五访问记录的相似用户
                pccsum += pcc;
                k+=1;
                prev+=pcc*(train[uid][s]-uMean[uid]);
                
            if pccsum==0:
                prev=uMean[i];
            else:
                prev = prev/pccsum +uMean[i];
            
            if prev<=0:prev=uMean[i];# 忽略预测值为负的值
            predict[i,s]=prev;
    return predict;


def ipcc(train,topK):
    predict = upcc(train.T.copy(),topK);
    return predict.T;

def UIPCC(upcc_predict,ipcc_predict,u_rate):
    return u_rate*upcc_predict+(1-u_rate)*ipcc_predict;


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
                predict[:,:,t] = upcc(train[:,:,t],10);
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

    pass