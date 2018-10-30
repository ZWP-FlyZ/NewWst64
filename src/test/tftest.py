# -*- coding: utf-8 -*-
'''
Created on 2018年8月4日

@author: zwp12
'''

import tensorflow as tf;



def run():
    x = tf.constant(3, tf.float32);
    rd = tf.random_normal((1,));
    y = x+rd;
    print(x);
    with tf.Session() as sess:
        print(sess.run(x));
        print(sess.run(rd));
        print(sess.run((x,rd,y)))
        pass;
    pass;

def model(x,y):
    

    out= tf.concat([x,y], axis=1);
    
    for i in range(1,3):
        out = tf.layers.dense(out,i,activation=tf.sigmoid);
    
    return out;

    
    
    pass;


if __name__ == '__main__':
    run();
    
#     x=[[1,2,3],[4,5,6]];
#     y=[[4,5,6],[1,2,3]];
#     xp = tf.placeholder(tf.float32, [None,3],"x");
#     yp = tf.placeholder(tf.float32, [None,3],"y");
#     rx = tf.random_normal([2,3]);
#     z = model(xp,yp);
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         print(sess.run(z,{xp:rx,yp:rx}));
    
    print(tf.__version__)
    
    
    pass