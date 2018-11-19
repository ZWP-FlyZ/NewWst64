# -*- coding: utf-8 -*-
'''
Created on 2018年10月17日

@author: zwp12
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
user_info_path=base_path+'/Dataset/ws/localinfo/user_info.txt';
ser_info_more_path=base_path+'/Dataset/ws/localinfo/ws_info_more.txt';
loc_class_out = base_path+'/Dataset/ws/localinfo/ws_classif_out.txt';


def run():
    user_loc = localload.load(user_info_path);
    
    rec = {};
    
    for loc in user_loc:
        name = user_loc[loc][1];
        rec[name] = rec.get(name,0)+1;
#         print(loc,user_loc[loc]);
    
    for name in rec:
        print(name,rec[name]);
    
    Y = [];
    sli = sorted(rec.items(), key=lambda p:p[1], reverse=True);
    for i,item in enumerate(sli):
        print(i,item);
        Y.append(item[1]);

    Y = np.array(Y);
    X = np.arange(len(Y));
    
    plt.bar(X, Y);
    
    plt.show();



if __name__ == '__main__':
    run();
    pass