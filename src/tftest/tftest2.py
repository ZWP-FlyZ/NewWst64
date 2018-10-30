# -*- coding: utf-8 -*-
'''
Created on 2018年10月30日

@author: zwp12
'''

import tensorflow as tf;
import numpy as np;
'''
测试获得重复参数

'''

ust_shape = [10,15,5];



def front(Pu,Qs,Rt,hid_units,regfunc=None):
    out = tf.concat([Pu,Qs,Rt],axis=1); 

    for i,unit in enumerate(hid_units):
        name = 'front_layer%d_unit%d'%(i,unit);
        out = tf.layers.dense(out, unit, 
                            activation=tf.nn.relu,
                            kernel_regularizer=regfunc,
                            name=name, reuse=tf.AUTO_REUSE);
    name = 'front_out';
    out=tf.layers.dense(inputs=out,units=1,
                        kernel_regularizer=regfunc,
                        name=name, reuse=tf.AUTO_REUSE);    
    return out;


def get_hid_var(shape,name,regfunc=None):
    with tf.variable_scope('hidden',reuse=tf.AUTO_REUSE):
        v = tf.get_variable(name,shape=shape,
                    dtype=tf.float32,
                    initializer=tf.initializers.ones,
                    regularizer=regfunc);
    return v;    

def run():
    print(tf.__version__);

#     regfunc = tf.contrib.layers.l2_regularizer(0.01);
    regfunc = tf.keras.regularizers.l2(0.01);
    
#     Pu = get_hid_var([2,2],'Pu',regfunc);
#     Qs = get_hid_var([2,2],'Qs',regfunc);
#     Rt = get_hid_var([2,2],'Rt',regfunc);
    
    Pu = tf.placeholder(tf.float32,[None,2]);
    Qs = tf.placeholder(tf.float32,[None,2]);
    Rt = tf.placeholder(tf.float32,[None,2]);
    Rt2 = tf.placeholder(tf.float32,[None,2]);
    print(Pu,Qs,Rt);
    
    out = front(Pu,Qs,Rt,[3,2],regfunc);
    print(Rt[:1]);
    out2 = front(Pu,Qs,Rt2,[3,2],regfunc);
    
    
    print(tf.losses.get_regularization_losses());
    tolosses = tf.losses.get_regularization_loss();
    print(tolosses);
    print(tf.trainable_variables())
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer());
        print(sess.run([Pu,Qs,Rt,out,out2,tolosses],
                       {Pu:np.ones([2,2]),
                        Qs:np.ones([2,2]),
                        Rt:np.ones([2,2]),
                        Rt2:np.zeros([2,2])}));
    
    


if __name__ == '__main__':
    run();
    pass