# -*- coding: utf-8 -*-
'''
Created on 2018年9月4日

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


base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work';
origin_path = base_path+'/Dataset/ws/rtmatrix.txt';
ser_info_path=base_path+'/Dataset/ws/ws_info.txt';
ser_info_more_path=base_path+'/Dataset/ws/ws_info_more.txt';
loc_class_out = base_path+'/Dataset/ws/ws_classif_out.txt';

def simple_km(data,k):
    datasize = len(data);
    if k<1 or  datasize<k:
        raise ValueError('data,k err');
    cents=data[random.sample(range(0,datasize),k)];
    last_c = cents;
    while True:
        res = [[] for _ in range(k)];
        for i in range(datasize):
            dis = np.abs(cents-data[i]);
            dis[:,1]=np.where(dis[:,1]>180.0,360.0-dis[:,1],dis[:,1]);
            dis = np.sum(dis**2,axis=1);
            tagk= np.argmin(dis);
            res[tagk].append(i);
        last_c = np.copy(cents);
        for i in range(k):
            cents[i]=np.mean(data[res[i]],axis=0);    
        bout = np.sum(cents-last_c);
        if bout==0:break;

    return cents,res;
    pass;


def classf(carr,tagdir):
    res = [];
    for idx in tagdir:
        if tagdir[idx][1] in carr:
            res.append(idx);
    fwrite.fwrite_append(loc_class_out, utils.arr2str(res));




def run():
    
    ser_loc = localload.load(ser_info_path);
    ser_loc_m = localload.load_locmore(ser_info_more_path);
    os.remove(loc_class_out);
    
    
    data=[];
    names=[];
    area=[];
    k=6;
    for sn in ser_loc_m:
        data.append(ser_loc_m[sn][1]);
        names.append(sn);
        area.append(ser_loc_m[sn][0])
    data=np.array(data);
    cent,res = simple_km(data,k);
    
    print(cent);
    print(res);
    
    for i in range(k):
        tmp=[];
        tmp2=[];
        for id in res[i]:
            tmp.append(area[id]);
            tmp2.append(names[id]);
        print(tmp)
        print(tmp2);
        print();
        classf(tmp2,ser_loc);   
    pass;

if __name__ == '__main__':
    run();
    pass

