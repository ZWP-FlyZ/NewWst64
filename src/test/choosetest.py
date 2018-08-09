# -*- coding: utf-8 -*-
'''
Created on 2018年8月9日

@author: zwp12
'''


import tensorflow as tf;


# shape=2*3*4
data = tf.constant([[[1,2,3,4],[5,6,7,8],[9,10,11,12]],
                    [[13,14,15,16],[17,18,19,20],[21,22,23,24]]]);

# index [dim1,dim2]->tensor dim3
idx = tf.constant([[0,0], # [1,2,3,4]
                   [1,0], #  [13,14,15,16]
                   [1,2]])# [21,22,23,24]


res = tf.gather_nd(data,idx);


data = tf.random_normal([10,3]);
ut = tf.concat([data[:,0:1],data[:,2:3]],axis=1);
st = data[:,1:3];


with tf.Session() as sess:
    dv,uv,sv = sess.run((data,ut,st));
    print(dv);
    print(uv);
    print(sv);



if __name__ == '__main__':
    pass