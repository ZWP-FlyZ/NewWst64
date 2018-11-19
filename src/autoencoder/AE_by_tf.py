# -*- coding: utf-8 -*-
'''
Created on 2018年11月19日

@author: zwp12
'''

'''
测试失败
'''

import tensorflow as tf;
import numpy as np;
import time;
from tools import SysCheck

class AE_TF():
    
    def __init__(self,hid_f):
        self.hid_f = hid_f;
        
    def model(self,X,reg_func=None):
        x_size = X.shape[1];
        hid_f = self.hid_f;
        
        enco = tf.layers.dense(X,hid_f,
                    activation=tf.tanh,
                    kernel_regularizer=reg_func);
        deco = tf.layers.dense(enco,x_size,
                    activation=tf.tanh,
                    kernel_regularizer=reg_func);
        
        py = deco;
        
        loss = tf.losses.huber_loss(X,py);
        mae = tf.reduce_mean(tf.abs(py-X));
        return py,loss,mae;
        
    def train(self,R,tR,lr,epoch,batch,reg_w = 0.01):
    
        reg_func = tf.keras.regularizers.l2(reg_w);
        global_step = tf.Variable(0,trainable=False,name='gs');
        # 准备数据集
        train_ds = tf.data.Dataset.from_tensor_slices(R);
        train_ds = train_ds.shuffle(100).batch(batch);
        test_ds = tf.data.Dataset.from_tensor_slices(R).batch(tR.shape[0]);
        it = tf.data.Iterator.from_structure(train_ds.output_types,
                                            train_ds.output_shapes);
        
        X = it.get_next(); 
        train_ds_op = it.make_initializer(train_ds);
        test_ds_op = it.make_initializer(test_ds);
                
        
        py,loss,mae = self.model(X,reg_func);
        
        train_step = tf.train.GradientDescentOptimizer(lr).minimize(loss, global_step)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer());
            eptime = time.time();
            for ep in range(epoch):
                print('ep%d 开始'%ep);
                
                # 训练阶段
                sess.run(train_ds_op);
                while True:
                    try:
                        now= time.time();
                        _,vloss,vmae,step= sess.run((train_step,loss,mae,global_step));
                        if step%5==0:
                            print('--->ep%d step%d time=%.2f loss=%f mae=%f'%(ep,step,time.time()-now,vloss,vmae));
                            now = time.time();            
                    except tf.errors.OutOfRangeError:
                        break;
                
                # 测试阶段
                sess.run(test_ds_op);
                while True:
                    try:
                        vpy,vloss= sess.run((py,loss));
                        vpy*=20.0;
                        delta = np.subtract(vpy,tR,out=np.zeros_like(tR),
                                            where=tR>0);
                        vmae = np.sum(np.abs(delta))/np.count_nonzero(delta);
                       
                    except tf.errors.OutOfRangeError:
                        break;
                                    
                eps = '==================================================\n'
                eps += 'ep%d结束 \t eptime=%.2f\n' %(ep ,time.time()-eptime);
                eps += 'test_mae=%f loss=%f\n'%(vmae,vloss);
                eps += 'acttime=%s\n'%(time.asctime());
                eps += '==================================================\n'
                eptime = time.time();
                print(eps);
        pass;

base_path = r'E:/work';
if SysCheck.check()=='l':
    base_path='/home/zwp/work'

us_shape=(339,5825);

spa=5
case=1;

isUAE = True;

hid_f = 100;
lr=0.01;
epoch=10;
batch = 10;
reg_w = 0.000001;

def run(spa,case):
    train_data = base_path+'/Dataset/ws/train_n/sparseness%.1f/training%d.txt'%(spa,case);
    test_data = base_path+'/Dataset/ws/test_n/sparseness%.1f/test%d.txt'%(spa,case);
    
    
    print('开始实验，稀疏度=%d,case=%d'%(spa,case));
    print ('加载训练数据开始');
    now = time.time();
    trdata = np.loadtxt(train_data, dtype=float);
    n = np.alen(trdata);
    print ('加载训练数据完成，耗时 %.2f秒，数据总条数%d  \n'%((time.time() - now),n));
    
    print ('转换数据到矩阵开始');
    tnow = time.time();
    u = trdata[:,0];
    s = trdata[:,1];
    u = np.array(u,int);
    s = np.array(s,int);
    R = np.full(us_shape, 0.0, float);
    R[u,s]=trdata[:,2];
    del trdata,u,s;
    print ('转换数据到矩阵结束，耗时 %.2f秒  \n'%((time.time() - tnow)));    
    
    print ('加载测试数据开始');
    now = time.time();
    trdata = np.loadtxt(test_data, dtype=float);
    n = np.alen(trdata);
    print ('加载测试数据完成，耗时 %.2f秒，数据总条数%d  \n'%((time.time() - now),n));
    
    print ('转换测试数据到矩阵开始');
    tnow = time.time();
    u = trdata[:,0];
    s = trdata[:,1];
    u = np.array(u,int);
    s = np.array(s,int);
    tR = np.full(us_shape, 0.0, float);
    tR[u,s]=trdata[:,2];
    del trdata,u,s;
    print ('转换测试数据到矩阵结束，耗时 %.2f秒  \n'%((time.time() - tnow)));    
    
    
    print ('初始化数据');
    tnow = time.time();
    idx = np.where(R<0);
    R[idx]=0;
    idx = np.where(tR<0);
    tR[idx]=0;    
    R = R/20.0;

    if not isUAE:
        R = R.T;
        tR =tR.T;
    print ('初始化数据结束耗时 %.2f秒  \n'%((time.time() - tnow)));
    
    
    print ('训练模型开始');
    ae_mode = AE_TF(hid_f);
    ae_mode.train(R, tR, lr, epoch, batch, reg_w);
    print ('训练模型结束 %.2f秒  \n'%((time.time() - tnow)));    
    
    pass;





if __name__ == '__main__':\
    run(spa,case);
    
