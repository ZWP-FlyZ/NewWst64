# -*- coding: utf-8 -*-
'''
Created on 2018年10月30日

@author: zwp12
'''

import tensorflow as tf;
import numpy as np;
from tensorflow import keras;
from tensorflow.keras.layers import Dense,LSTM;


def run():
    
    mode = keras.Sequential();
    mode.add(Dense(5));
    
    
    mode.compile(optimizer=tf.train.AdamOptimizer(0.001),
                loss=tf.losses.mean_squared_error);
    
    
    data = np.array([[1,0,0,0,0],
                     [0,1,0,0,0],
                     [0,0,1,0,0],
                     [0,0,0,1,0],
                     [0,0,0,0,1]],np.float);
                     
    labels =np.array([[0,1,0,0,0],
                     [0,0,1,0,0],
                     [0,0,0,1,0],
                     [0,0,0,0,1],
                     [1,0,0,0,0]],np.float);
    
    mode.fit(data, labels, epochs=10, batch_size=1);
                
                
    pass;


if __name__ == '__main__':
    run();
    pass