# -*- coding: utf-8 -*-
'''
Created on 2018年10月17日

@author: zwp12
'''



import numpy as np;
import time;
from tools import SysCheck;
from tools import localload;

base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work';
origin_path = base_path+'/Dataset/ws/rtmatrix.txt';
user_info_path=base_path+'/Dataset/ws/localinfo/user_info.txt';
ser_info_more_path=base_path+'/Dataset/ws/localinfo/ws_info_more.txt';
loc_class_out = base_path+'/Dataset/ws/localinfo/ws_classif_out.txt';


def run():
    user_loc = localload.load_location_name(user_info_path);
    for loc in user_loc:
        print(loc)


if __name__ == '__main__':
    run();
    pass