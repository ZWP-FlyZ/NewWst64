# -*- coding: utf-8 -*-
'''
Created on 2018年9月13日

@author: zwp12
'''


import tensorflow as tf;
import numpy as np;

us_shape=[5,10];

a = [[0,2,3,4],
     [1,3,5,-1],
     [6,7,9,-1],
     [2,8,-1,-1],
     [3,5,7,-1]];
ca = [4,3,3,2,3];
a = np.array(a)+1;
ca = (1.0/np.sqrt(np.array(ca))).astype(np.float32)
print(a);
print(ca);

# def prf(bat,a):
# #     idx = tf.cast(bat[0],tf.int32);
# #     hot = tf.one_hot(a[idx],us_shape[1]);
# #     hot = tf.reduce_sum(hot,axis=0);
# #     print(hot);
# #     return hot;
#     print(bat);
#     return bat;
# 
data = tf.constant([[0,2,0.1],
                    [1,3,0.2],
                    [2,7,0.3]],dtype=tf.float32);
 
data = tf.reshape(data,[-1,3]);
# 
# ma = tf.map_fn(fn=lambda b:prf(b,a)
#                , elems=data, 
#                dtype=tf.float32);

y = tf.get_variable('y', [us_shape[1],3], dtype=tf.float32,
                    initializer=tf.initializers.truncated_normal());



n=3;
i=tf.constant(0);

res = tf.placeholder(tf.float32, [None,us_shape[1]+1])
# res = tf.constant(0,tf.float32,[1,us_shape[1]+1]);
def cond(i,_,data,a):
    return i<n;

def body(i,res,data,a):
    idx = tf.cast(data[i,0],tf.int32);
    print(idx);
    hot = tf.one_hot(tf.gather(a,idx)
                     ,us_shape[1]+1);
    hot = tf.reduce_sum(hot,axis=0,keepdims=True)*tf.gather(ca,idx);
    
    res = tf.cond(tf.equal(i,tf.constant(0)),
                  lambda:hot,
                  lambda:tf.concat([res,hot],axis=0));
    return tf.add(i,1),res,data,a;

_,out,_,_ = tf.while_loop(cond,body,[i,res,data,a]);

print(out.shape.as_list())

out = tf.slice(out,[0,1],[n,us_shape[1]]);

ret = tf.matmul(out,y);

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer());
    yv,outv,retv=sess.run([y,out,ret],{res:[[0]*(us_shape[1]+1)]});
    print(yv);
    print(outv);
    print(retv);

if __name__ == '__main__':
    pass