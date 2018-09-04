# -*- coding: utf-8 -*-
'''
Created on 2018年9月4日

@author: zwp12
'''


import numpy as np;
import random;

def simple_km(data,k):
    datasize = len(data);
    if k<1 or  datasize<k:
        raise ValueError('data,k err');
    cents=data[random.sample(range(0,datasize),k)];
    last_c = cents;
    while True:
        res = [[] for _ in range(k)];
        for i in range(datasize):
            dis = np.sum((cents-data[i])**2,axis=1);
            tagk= np.argmin(dis);
            res[tagk].append(i);
        last_c = np.copy(cents);
        for i in range(k):
            cents[i]=np.mean(data[res[i]],axis=0);    
        bout = np.sum(cents-last_c);
        if bout==0:break;

    return cents,res;
    pass;


def run():
    data = np.array([[0,0],
                     [0,1],
                     [1,0],
                     [2,1],
                     [2,0]],dtype=np.float)
    
    cent,res = simple_km(data,1);
    print(cent);
    print(res);
    for ids in res:
        print(data[ids]);
    
    pass;

if __name__ == '__main__':
    run();
    pass



