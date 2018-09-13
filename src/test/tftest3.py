# -*- coding: utf-8 -*-
'''
Created on 2018年9月13日

@author: zwp12
'''


import tensorflow as tf;
import numpy as np;


v = [[0,1,1,0],[1,0,0,1],[1,1,1,0],[0,0,0,1]]
v = tf.constant(v,tf.int32);
print(v);

data = tf.constant([[0,2,0.1],
                    [1,3,0.2],
                    [2,7,0.3]],dtype=tf.float32);
idx = tf.reshape(tf.cast(data[:,0],tf.int32),[-1,1]);
print(idx);
vout = tf.gather(v,idx);
                    
                    
with tf.Session() as sess:
    print(sess.run([data[:,1],vout]));

