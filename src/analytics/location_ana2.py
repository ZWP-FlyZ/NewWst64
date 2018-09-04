# -*- coding: utf-8 -*-
'''
Created on 2018年9月4日

@author: zwp12
'''

'''
    原始数据在服务地理位置分布
'''

import numpy as np;
import time;
from tools import SysCheck;
from tools import localload;

base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work';
origin_path = base_path+'/Dataset/ws/rtmatrix.txt';
ser_info_path=base_path+'/Dataset/ws/ws_info.txt';
ser_info_more_path=base_path+'/Dataset/ws/ws_info_more.txt';


def run():
    ser_loc = localload.load(ser_info_path);
    ser_loc_m = localload.load_locmore(ser_info_more_path);
    ser_name = localload.load_location_name(ser_info_path);
    oridata = np.loadtxt(origin_path,np.float);
    _,s = np.where(oridata);
    res={};
    res1={};
    for sn in ser_name:
        res[sn]=0;
    res1['SA']=0;
    res1['A']=0;
    res1['O']=0;
    res1['E']=0;
    res1['NA']=0;
    res1['ME']=0;
    res1['SAF']=0;
    
    
    for sid in s:
        sn = ser_loc[sid][1];
        res[sn]=res[sn]+1;
        an = ser_loc_m[sn][0];
        res1[an]=res1[an]+1;
    
    
    for i in res:
        print(i,res[i]);
    print();
    for i in res1:
        print(i,res1[i]);    
    pass;


if __name__ == '__main__':
    run();
    pass