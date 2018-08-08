# -*- coding: utf-8 -*-
'''
Created on 2018年8月8日

@author: zwp12
'''

import numpy as np;
import time;

# 矩阵相乘方法
# a=  np.random.normal(size=(4500,64,32))
# 
# b = np.random.normal(size=(100,4500,1,64));
# print(a);
# print(np.matmul(b,a));

# ba = np.matmul(b,a);
# print(ba)
# print(ba.shape)

a = np.arange(24).reshape((4,3,2));
print(a);
b = np.zeros((1,1,3,2))
b[0,0,0,0]=1;

res = b*a;


print(res);

t = np.sum(res,axis=(2,3));
print(t);







if __name__ == '__main__':
    pass