# -*- coding: utf-8 -*-
'''
Created on 2018年10月19日

@author: zwp12
'''


'''
用户地理位置与服务地理位置联合聚类分析

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
user_info_path=base_path+'/Dataset/ws/localinfo/user_info.txt';
ser_info_more_path=base_path+'/Dataset/ws/localinfo/ws_info_more.txt';
loc_class_out = base_path+'/Dataset/ws/localinfo/ws_classif_out.txt';

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
    data[:,2]=data[:,2]/di;
    data[:,3]=data[:,3]/(di*2.0);    
    cents=data[random.sample(range(0,datasize),k)];
    last_c = cents;
    
    while True:
        res = [[] for _ in range(k)];
        for i in range(datasize):
            dis = np.abs(cents-data[i]);
            dis[:,1]=np.where(dis[:,1]>edg,edg2-dis[:,1],dis[:,1]);
            dis[:,3]=np.where(dis[:,3]>edg,edg2-dis[:,3],dis[:,3]);
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
        

    return cents,res;
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
    user_loc = localload.load_userinfo(user_info_path);
    ser_loc_m = localload.load_locmore(ser_info_more_path);
    user_class = localtools.load_classif(loc_class_for_user);
    R = np.loadtxt(origin_path,np.float);
    
#     os.remove(loc_class_out);

    idx = np.where(R>0);
    
    u = idx[0].astype(np.int);
    s = idx[1].astype(np.int);
    dataize = len(u);
    
    data=[];
    names=[];
    area=[];
    k=8;
    for sid in range(5825):
        sn = ser_loc[sid][1];
        names.append(sn);
        area.append(ser_loc_m[sn][0])
    
    for did in range(dataize):
        sn = ser_loc[s[did]][1];
        cl=[];
        cl.extend(user_loc[u[did]][2]);
        cl.extend(ser_loc_m[sn][1]);
        cl.append(R[u[did],s[did]]);
        data.append(cl);
    data= np.array(data);
    
    cent,res = simple_km(data,k,6);
    
    print(cent);
#     print(res);
    
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
        
    write2file(res);   
    pass;

if __name__ == '__main__':
    run();
    pass