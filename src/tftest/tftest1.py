# -*- coding: utf-8 -*-
'''
Created on 2018年10月30日

@author: zwp12
'''

import tensorflow as tf;

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
    
    out=tf.layers.dense(inputs=out,units=1,
                        kernel_regularizer=regfunc);    
    return out;


def get_hid_var(shape,name,regfunc=None):
    with tf.variable_scope('hidden',reuse=tf.AUTO_REUSE):
        v = tf.get_variable(name,shape=shape,
                    dtype=tf.float32,
                    initializer=tf.initializers.zeros,
                    regularizer=regfunc);
    return v;    

def run():
    print(tf.__version__);

#     regfunc = tf.contrib.layers.l2_regularizer(0.01);
    regfunc = tf.keras.regularizers.l2(0.01);
    v1 = get_hid_var([2,2],'P',regfunc);
    v2 = get_hid_var(None,'P',regfunc);
    g3 = tf.layers.dense(v1,3,
        kernel_regularizer=regfunc,
        name='Q',
        reuse=tf.AUTO_REUSE
        );
    Q = tf.trainable_variables('Q/kernel')[0];
    print(Q);
#     for it in Q:
#         print(it,it.name);
    q = Q[:1];
    
    
    print(v1,v2,g3);
#     assert v1==v2;
    print(tf.losses.get_regularization_losses());
    tolosses = tf.losses.get_regularization_loss();
    print(tolosses);
    
    print(tf.trainable_variables())
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer());
        print(sess.run([v1,v2,Q,q,tolosses]));
    



def run2():
    
    
    a = tf.constant([[1,2,3],[3,4,5],[7,8,9]]);
    b = -a;
    ta = tf.concat([a[:2],b[2:3], a[3:4]],axis=0);
    print(a.shape,b,ta);
    
    
    with tf.Session() as sess:
        av,bv,tav=sess.run([a,b,ta]);
        
    
        print(av);
        print(bv);
        print(tav);



if __name__ == '__main__':
    run2();
    pass