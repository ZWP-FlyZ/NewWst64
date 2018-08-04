# -*- coding: utf-8 -*-
'''
Created on 2018年8月4日

@author: zwp12
'''

'''
WSDream 64 时间数据集随机分割方案

'''


import time;
import numpy as np;
import random;
import os;
from tools import SysCheck;

base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'
    
origin_data_path = base_path+'/Dataset/wst64/rtdata.txt';
train_output_path = base_path+'/Dataset/wst64/train_n'
test_output_path = base_path+'/Dataset/wst64/test_n'

# 分割稀疏度列表
spa_list=[];

# 每个稀疏度需要的例子数
case_cout=10;


# 加载数据集
def load_origin_data(path):
    rec_table = np.loadtxt(path,float); 
    pass;

def save_data(path,data):
    np.savetxt(path,data,'%d %d %d %.3f');
    pass;




if __name__ == '__main__':
    pass