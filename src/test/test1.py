# -*- coding: utf-8 -*-
'''
Created on 2018年8月4日

@author: zwp12
'''


import numpy as np;

a = np.arange(9).reshape((3,3));
b = np.array([[0,0],
              [1,1],
              [2,2]]);
b[:,1] = np.where(b[:,1]>0,2-b[:,1],b[:,1]);

print(b);
if __name__ == '__main__':
    pass