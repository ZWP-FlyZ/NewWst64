# -*- coding: utf-8 -*-
'''
Created on 2018年10月31日

@author: zwp12
'''



import tensorflow as tf;
import numpy as np;
from tensorflow import keras;
from tensorflow.keras.layers import Dense,LSTM;



class Model(tf.keras.Model):
    def __init__(self, hid_size, units):
        super(Model, self).__init__()
        self.units = units;
    
        self.gru = tf.keras.layers.GRU(self.units, 
                                     return_sequences=True, 
                                     recurrent_activation='sigmoid', 
                                     recurrent_initializer='glorot_uniform', 
                                     stateful=True)
    
        self.fc = tf.keras.layers.Dense(hid_size)
            
    def call(self, x):
      
        # output at every time step
        # output shape == (batch_size, seq_length, hidden_size) 
        output = self.gru(x)
        
        # The dense layer will output predictions for every time_steps(seq_length)
        # output shape after the dense layer == (seq_length * batch_size, vocab_size)
        prediction = self.fc(output)
        
        # states will be used to pass at every step to the model while training
        return prediction    


def split(item):
    return item[:-1],item[1:];


def train(R,seq_len,unit):
    t_s,hid_size = R.shape;
#     splitedR = tf.split(R, num_or_size_splits=cot, axis=0)
    splitedR = tf.reshape(R,[-1,seq_len+1,hid_size]);
    print(splitedR);
    ds = tf.data.Dataset.from_tensor_slices(splitedR);
#     ds.map(lambda i:(i[:-1],i[1:]));
    it = ds.make_initializable_iterator();
    nextitem = it.get_next();
    x,y=tf.expand_dims(nextitem[:-1],0),\
            tf.expand_dims(nextitem[1:],0);
    
    x = tf.cast(x,tf.float32);
    geu = Model(hid_size,unit);
     
    py = geu(x);
    
    loss = tf.reduce_mean(tf.losses.mean_squared_error(y,py));
    mae = tf.reduce_mean(tf.abs(y-py));
    
    train_op = tf.train.AdamOptimizer(0.007).minimize(loss);
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for ep in range(200):
            print('ep%d\n'%ep);
            sess.run(it.initializer)
            maesum=0.0;
            cot=0;
            while True:
                try:
                    _,vx,vy,vpy,vloss,vmae=sess.run([train_op,x,y,py,loss,mae])
                    maesum+=vmae;
                    cot+=1;
#                     print('x:\n',vx);
#                     print('y:\n',vy);
#                     print('py:\n',vpy);
                    print('loss:\n',vloss);
                    print('mae:\n',vmae);
                    print('------------------------')
                except tf.errors.OutOfRangeError:break
            print('\n------------>',maesum/cot);
            
                    
            

def run():
    
#     mode = keras.Sequential();
#     mode.add(Dense(5));
#     
#     
#     mode.compile(optimizer=tf.train.AdamOptimizer(0.001),
#                 loss=tf.losses.mean_squared_error);
#     
#     
#     data = np.array([[1,0,0,0,0],
#                      [0,1,0,0,0],
#                      [0,0,1,0,0],
#                      [0,0,0,1,0],
#                      [0,0,0,0,1]],np.float);
#                      
#     labels =np.array([[0,1,0,0,0],
#                      [0,0,1,0,0],
#                      [0,0,0,1,0],
#                      [0,0,0,0,1],
#                      [1,0,0,0,0]],np.float);
#     
#     mode.fit(data, labels, epochs=10, batch_size=1);    
    
    d = np.random.normal(size=(20,32));
#     d = np.arange(-4,4,0.1,dtype=np.float).reshape([20,4])
    print(d);
    data = tf.constant(d,dtype=tf.float32);
    train(data,3,10);
    
    
    pass;

if __name__ == '__main__':
    run();
    pass