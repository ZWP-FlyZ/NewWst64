# -*- coding: utf-8 -*-
'''
Created on 2018年11月1日

@author: zwp12
'''



import numpy as np;
import time;
import random;
import os;
from tools import SysCheck;
from tools import localload;
from tools import utils;
from tools import fwrite;
from location import localtools

base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work';
origin_path = base_path+'/Dataset/ws/rtmatrix.txt';
ser_info_path=base_path+'/Dataset/ws/localinfo/ws_info.txt';
ser_info_more_path=base_path+'/Dataset/ws/localinfo/ws_info_more.txt';
loc_class_out = base_path+'/Dataset/ws/localinfo/ws_classif_out.txt';
loc_class_dis_rate_out = base_path+'/Dataset/ws/localinfo/ws_classif_dr_out.txt';

loc_class_for_user = base_path+'/Dataset/ws/localinfo/ws_classif_out_by_user.txt';


def simple_km(data,k,di=1.0):
    datasize = len(data);
    di = float(di);
    if k<1 or  datasize<k:
        raise ValueError('data,k err');
    rep=0;
    edg=180.0/di;
    edg2=  edg*2;
    data[:,0]=data[:,0]/di;
    data[:,1]=data[:,1]/(di*2.0);
    cents=data[random.sample(range(0,datasize),k)];
    last_c = cents;
    
    while True:
        res = [[] for _ in range(k)];
        for i in range(datasize):
            dis = np.abs(cents-data[i]);
            dis[:,1]=np.where(dis[:,1]>edg,edg2-dis[:,1],dis[:,1]);
            dis = np.sum(dis**2,axis=1);
            tagk= np.argmin(dis);
            res[tagk].append(i);
        last_c = np.copy(cents);
        for i in range(k):
            cents[i]=np.mean(data[res[i]],axis=0);    
        bout = np.sum(cents-last_c);
        if bout==0:break;
        rep+=1;
        if rep%1 == 0:
            print('rep=%d,delta=%f'%(rep,bout));
        
#     cents[:,0] *= di;
#     cents[:,1] *= 2*di;
    dis_li= [];
    for i in range(k):
        dis = np.abs(data-cents[i]);
        dis[:,1]=np.where(dis[:,1]>edg,edg2-dis[:,1],dis[:,1]);
        dis = np.sqrt(np.sum(dis**2,axis=1));
        dis_li.append(dis);
        
    return cents,res,np.array(dis_li);
    pass;


def classf(carr,tagdir):
    res = [];
    for idx in tagdir:
        if tagdir[idx][1] in carr:
            res.append(idx);
    fwrite.fwrite_append(loc_class_out, utils.arr2str(res));


def write2file(res):
    for li in res:
        fwrite.fwrite_append(loc_class_out, utils.arr2str(li));



def run():
    
    ser_loc = localload.load(ser_info_path);
    ser_loc_m = localload.load_locmore(ser_info_more_path);
    user_class = localtools.load_classif(loc_class_for_user);
    R = np.loadtxt(origin_path,np.float);
    
    

    idx = np.where(R<0);
    R[idx]=0;
    user_mean = [];
    for uc in user_class:
        UR= R[uc];
        ser_sum = np.sum(UR,axis=0);
        ser_cot = np.count_nonzero(UR, axis=0);
        uc_ser_mean = np.divide(ser_sum,ser_cot,
            out=np.zeros_like(ser_sum),where=ser_cot!=0);
        all_mean = np.sum(ser_sum)/np.sum(ser_cot);
        uc_ser_mean[np.where(ser_cot==0)] = all_mean;
        
        user_mean.append(uc_ser_mean);
    
    data=[];
    names=[];
    area=[];
    k=6;
    di = 3;
    for sid in range(5825):
        sn = ser_loc[sid][1];
        names.append(sn);
        area.append(ser_loc_m[sn][0])
        lc = [];
        lc.extend(ser_loc_m[sn][1]);
        for um in user_mean:
            lc.append(um[sid]);
        data.append(lc);
    data=np.array(data);
#     np.random.shuffle(data);
    cent,res,dis_rate = simple_km(data,k,di);
    
    print(cent);
    print(res);
    
    for i in range(k):
        tmp=[];
        tmp2=[];
        for id in res[i]:
            if names[id] not in tmp2:
                tmp2.append(names[id]);
                tmp.append(area[id]);
        print(tmp)
        print(tmp2);
        print();
    
    
    # 计算类别距离

    
    dis_rate = 1 / dis_rate;
    print(dis_rate);
    print(np.sort(dis_rate,axis=1));
    
    dis_rate = np.exp(dis_rate);
    dis_sum = np.sum(dis_rate,axis=0);
    dis_rate /= dis_sum;
    
    print(dis_rate);
    
    print(np.sort(dis_rate,axis=1));
    
    np.savetxt(loc_class_dis_rate_out,dis_rate,'%.8f')
    
    os.remove(loc_class_out);    
    write2file(res);   
    pass;

if __name__ == '__main__':
    run();
    pass