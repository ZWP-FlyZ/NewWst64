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

a = np.arange(64).reshape((4,4,4));
print(a);
u = np.arange(4).reshape([4,1]);
s = np.arange(4).reshape([4,1]);
ind = np.arange(4).reshape([4,1]);


print(a[ind,u,s])














if __name__ == '__main__':
    pass