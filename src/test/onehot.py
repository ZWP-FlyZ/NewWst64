# -*- coding: utf-8 -*-
'''
Created on 2018年8月8日

@author: zwp
'''
import tensorflow as tf;
import numpy as np;


data = np.random.randint(0,9,size=(10,3));
print(data);

feature = tf.convert_to_tensor(data,dtype=tf.int32);

print(feature);

sh=tf.shape(feature);
  
UT = tf.zeros([sh[0],10,10]);
ST = tf.zeros([sh[0],10,10]);
U = tf.reshape(feature[:,0],[-1]);
S=tf.reshape(feature[:,1],[-1]);
T=tf.reshape(feature[:,2],[-1]);
ind = tf.convert_to_tensor(tf.range(sh[0]));
ind = tf.cast(ind,tf.int32);
print(ind);
e1 = (ind,U,T);
e2 = ind,S,T;
indut = tf.map_fn(fn=lambda x: [x[0],x[1],x[2]],
          elems=e1,dtype=[tf.int32,tf.int32,tf.int32])
indst = tf.map_fn(fn=lambda x: [x[0],x[1],x[2]],
          elems=e2,dtype=[tf.int32,tf.int32,tf.int32])



indst = tf.convert_to_tensor(indst);
indst = tf.reshape(indst,[-1,3]);

print(indut);
print(indst);
spav = tf.ones([sh[0]],dtype=tf.int32);
print(spav);


with tf.Session() as sess:
    print(sess.run(e1));
    print(sess.run((indut)));



if __name__ == '__main__':
    pass