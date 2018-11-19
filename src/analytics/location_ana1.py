# -*- coding: utf-8 -*-
'''
Created on 2018年9月4日

@author: zwp12
'''

'''
    服务地理位置分布
'''

import numpy as np;
import time;
from tools import SysCheck;
from tools import localload;

import matplotlib.pyplot as plt



base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work';
origin_path = base_path+'/Dataset/ws/rtmatrix.txt';
ser_info_path=base_path+'/Dataset/ws/localinfo/ws_info.txt';
ser_info_more_path=base_path+'/Dataset/ws/localinfo/ws_info_more.txt';
loc_class_out = base_path+'/Dataset/ws/localinfo/ws_classif_out.txt';


def run():
    ser_loc = localload.load(ser_info_path);
    ser_loc_m = localload.load_locmore(ser_info_more_path);
    
    res={};
    res1={};
    for serid in ser_loc:
        sn = ser_loc[serid][1];
        if sn not in res:
            res[sn]=0;
        res[sn]=res[sn]+1;
        an = ser_loc_m[sn][0];
        if an not in res1:
            res1[an]=0;
        res1[an]=res1[an]+1;
    
    res = sorted(res.items(), key=lambda p:p[1], reverse=True)
        

    Y = [];
    for i,item in enumerate(res):
        print(i,item);
        Y.append(item[1]);
    
    Y = np.array(Y);
    X = np.arange(len(Y));
    
    plt.bar(X, Y);
    plt.ylabel('number of services');
    plt.show();
        
        
        
    print();
#     for i in res1:
#         print(i,res1[i]);    
#     pass;


if __name__ == '__main__':
    run();
    pass