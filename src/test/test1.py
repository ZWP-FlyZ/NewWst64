# -*- coding: utf-8 -*-
'''
Created on 2018年8月4日

@author: zwp12
'''


import numpy as np;

a = np.arange(9).reshape((3,3));
b = np.array([[0,0,0],
              [1,1,-1],
              [2,2,2]]);

print(a);
print(b.shape);
              
print(np.dot(a,b));

ind = np.where(b[:,2]>0)[0]
print(b[ind,:])

if __name__ == '__main__':
    pass