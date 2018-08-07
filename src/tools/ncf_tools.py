# -*- coding: utf-8 -*-
'''
Created on 2018年8月7日

@author: zwp12
'''

import numpy as np;

def del_non_data(rec_table):
    
    ind = np.where(rec_table[:,2]>0);
    return rec_table[ind,:][0];

def reoge_data(rec_table):
    rec_table = np.array(rec_table);
    rec_table = del_non_data(rec_table);
    return rec_table[:,0:2].astype(int),    \
        rec_table[:,2].reshape((-1,1)).astype(np.float32)
    

if __name__ == '__main__':
    
    a = np.array([[1,2,3],[2,3,4],[0,3,-1]])
    print(reoge_data(a));
    
    
    
    pass