# -*- coding: utf-8 -*-
'''
Created on 2018年8月7日

@author: zwp12
'''

# from new_ncf.ncf_param import NcfCreParam3D,NcfTraParm3D;
import tensorflow as tf;
import time;
from tools.ncf_tools import reoge_data,reoge_data3D;
from tools.fwrite import fwrite_append;
from tools.tfboard_sum import var_summaries;
class hyb_ncf():
    '''
    >混合模型
    '''
    # 用户服务输入
    feature=None;# tf.placeholer=[[u,s],[u,s]......];
    # 响应时间 
    label=None; # [[rt],[rt]...]
    
    # 创建参数
    create_param = None;
    
    # 预测结果
    Py = None;
    # 损失
    loss = None;
    
    # 评测结果
    mae = None;
    rmse= None;
    
    # 数据迭代器
    data_iter=None;
    
    def __init__(self,NcfCreParam):
        self.uNum,self.sNum=NcfCreParam.us_shape;
        self.create_param = NcfCreParam;
#         self.feature = \
#             tf.placeholder(tf.float32, [None,self.uNum], 'feature');
#         self.label   = \
#             tf.placeholder(tf.float32, [None,self.sNum], 'label');
        pass;
    
    
    def toOneHot(self,feature):
        U = tf.one_hot(feature[:,0], self.uNum)
        S = tf.one_hot(feature[:,1],self.sNum);
        return U,S;
    
    def create_model(self,feat,Y,NcfCreParam):
        hid_f = NcfCreParam.hid_feat;                          
        hid_units = NcfCreParam.hid_units;
        
        hid_actfunc =tf.nn.relu;
        # out_actfunc = tf.sigmoid;

        reg_func = tf.contrib.layers.l2_regularizer(NcfCreParam.reg_p);
         

        U,S = self.toOneHot(feat);
        
        # 初始化 隐含特征矩阵
        Pu = tf.layers.dense(inputs=U,units=hid_f,
                           #  kernel_initializer=initializers.random_normal(stddev=2),
                            kernel_regularizer=reg_func);
        Qs = tf.layers.dense(inputs=S,units=hid_f,
                             # kernel_initializer=initializers.random_normal(stddev=2),
                            kernel_regularizer=reg_func);
        mPu= tf.layers.dense(inputs=U,units=hid_f,
                           #  kernel_initializer=initializers.random_normal(stddev=2),
                            kernel_regularizer=reg_func);
        mQs= tf.layers.dense(inputs=S,units=hid_f,
                            # kernel_initializer=initializers.random_normal(stddev=2),
                            kernel_regularizer=reg_func);
        # 传统矩阵分解                    
        mout= mPu * mQs;

        out = tf.concat([Pu,Qs],axis=1);        
        
        for unit in hid_units:
            out=tf.layers.dense(inputs=out,units=unit,
                                activation=hid_actfunc,
                                kernel_regularizer=reg_func);
            out=tf.layers.dropout(out,NcfCreParam.drop_p);                    
        
        # 双模型混合                        
        out =  tf.concat([mout,out],axis=1);
        print(out)
        # 输出层                       
        out=tf.layers.dense(inputs=out,units=1,
                            activation=hid_actfunc,
                            kernel_regularizer=reg_func);
        
        Py=out;                    
        # 误差                   
        loss = tf.reduce_mean(tf.losses.huber_loss(Y,out));
        # 评测误差
        mae = tf.reduce_mean(tf.abs(Y-out));
        rmse = tf.sqrt(tf.reduce_mean((Y-out)**2));
        return Py,loss,mae,rmse; 
    ############################# end  ##################################    
    
    
    def train(self,NcfTraParm):

        
        train_data = reoge_data(NcfTraParm.train_data);
        test_data = reoge_data(NcfTraParm.test_data);
        testn = len(test_data[0]);
        global_step = tf.Variable(0,trainable=False,name='gs');
        ds = tf.data. \
                Dataset.from_tensor_slices(train_data);
        ds = ds.shuffle(1000).batch(NcfTraParm.batch_size);
        
        test_ds = tf.data.Dataset.from_tensor_slices(test_data);
        test_ds = test_ds.batch(testn);
        it = tf.data.Iterator.from_structure(ds.output_types,
                                            ds.output_shapes);
        
        feat,Y = it.get_next(); 
        train_init_op = it.make_initializer(ds);
        test_init_op = it.make_initializer(test_ds);   
            
        Py,loss,tmae,trmse= self.create_model(feat, Y, self.create_param);
        
        # loss+=tf.losses.get_regularization_loss;
        
        lr = tf.train.exponential_decay(NcfTraParm.learn_rate, global_step,
                                NcfTraParm.lr_decy_step,
                                NcfTraParm.lr_decy_rate,
                                staircase=True);
                                
        train_step = tf.train.AdagradOptimizer(lr). \
                    minimize(loss, global_step );
        
        
        save = tf.train.Saver();
        with tf.Session() as sess:
            if NcfTraParm.load_cache_rec:
                save.restore(sess,NcfTraParm.cache_rec_path);
            else:
                sess.run(tf.global_variables_initializer()); 
            
            now = time.time();
            for ep in range(NcfTraParm.epoch):
                sess.run(train_init_op);
                while True:
                    try:
                        _,vloss,gs=sess.run((train_step,loss,global_step));
                        if gs%(500) == 0:
                            print('ep%d\t loopstep:%d\t time:%.2f\t loss:%f'%(ep,gs,time.time()-now,vloss))
                            now=time.time();
                    except tf.errors.OutOfRangeError:
                        print('to end');
                        break  
                sess.run(test_init_op);
                vmae,vrmse,vloss=sess.run((tmae,trmse,loss)); 
                print('ep%d结束 \t eponloss=%f\t test_mae=%f test_rmse=%f\n'%(ep ,vloss,vmae,vrmse));
            
            if NcfTraParm.cache_rec_path != '':
                save.save(sess,NcfTraParm.cache_rec_path)
        pass;
    
    pass;




###########################################




class hyb_ncf_rep():
    '''
    >混合模型
    '''
    # 用户服务输入
    feature=None;# tf.placeholer=[[u,s],[u,s]......];
    # 响应时间 
    label=None; # [[rt],[rt]...]
    
    # 创建参数
    create_param = None;
    
    # 预测结果
    Py = None;
    # 损失
    loss = None;
    
    # 评测结果
    mae = None;
    rmse= None;
    
    # 数据迭代器
    data_iter=None;
    
    def __init__(self,NcfCreParam):
        self.uNum,self.sNum=NcfCreParam.us_shape;
        self.create_param = NcfCreParam;
#         self.feature = \
#             tf.placeholder(tf.float32, [None,self.uNum], 'feature');
#         self.label   = \
#             tf.placeholder(tf.float32, [None,self.sNum], 'label');
        pass;
    
    
    def toOneHot(self,feature):
        U = tf.one_hot(feature[:,0], self.uNum)
        S = tf.one_hot(feature[:,1],self.sNum);
        return U,S;
    
    def create_model(self,feat,Y,NcfCreParam):
        hid_f = NcfCreParam.hid_feat;                          
        hid_units = NcfCreParam.hid_units;
        
        hid_actfunc =tf.nn.relu;
        # out_actfunc = tf.sigmoid;

        reg_func = tf.contrib.layers.l2_regularizer(NcfCreParam.reg_p);
         

        U,S = self.toOneHot(feat);
        
        # 初始化 隐含特征矩阵
        Pu = tf.layers.dense(inputs=U,units=hid_f,
                           #  kernel_initializer=initializers.random_normal(stddev=2),
                            kernel_regularizer=reg_func);
        Qs = tf.layers.dense(inputs=S,units=hid_f,
                             # kernel_initializer=initializers.random_normal(stddev=2),
                            kernel_regularizer=reg_func);
        mPu= tf.layers.dense(inputs=U,units=hid_f,
                           #  kernel_initializer=initializers.random_normal(stddev=2),
                            kernel_regularizer=reg_func);
        mQs= tf.layers.dense(inputs=S,units=hid_f,
                            # kernel_initializer=initializers.random_normal(stddev=2),
                            kernel_regularizer=reg_func);
  
        
        # 传统矩阵分解                    
        mout= mPu * mQs;

        out = tf.concat([Pu,Qs],axis=1);        
        
        for unit in hid_units:
            out=tf.layers.dense(inputs=out,units=unit,
                                activation=hid_actfunc,
                                kernel_regularizer=reg_func);
            out=tf.layers.dropout(out,NcfCreParam.drop_p);                    
        
        # 双模型混合                        
        out =  tf.concat([mout,out],axis=1);
        print(out)
        # 输出层                       
        out=tf.layers.dense(inputs=out,units=1,
                            activation=hid_actfunc,
                            kernel_regularizer=reg_func);
        
        Py=out;                    
        # 误差                   
        loss = tf.reduce_mean(tf.losses.huber_loss(Y,out));
        # 评测误差
        mae = tf.reduce_mean(tf.abs(Y-out));
        rmse = tf.sqrt(tf.reduce_mean((Y-out)**2));
        return Py,loss,mae,rmse; 
    ############################# end  ##################################    
    
    
    def train(self,NcfTraParm):

        
        train_data = reoge_data(NcfTraParm.train_data);
        test_data = reoge_data(NcfTraParm.test_data);
        testn = len(test_data[0]);
        global_step = tf.Variable(0,trainable=False,name='gs');
        ds = tf.data. \
                Dataset.from_tensor_slices(train_data);
        ds = ds.shuffle(1000).batch(NcfTraParm.batch_size);
        
        test_ds = tf.data.Dataset.from_tensor_slices(test_data);
        test_ds = test_ds.batch(testn);
        it = tf.data.Iterator.from_structure(ds.output_types,
                                            ds.output_shapes);
        
        feat,Y = it.get_next(); 
        train_init_op = it.make_initializer(ds);
        test_init_op = it.make_initializer(test_ds);   
            
        Py,loss,tmae,trmse= self.create_model(feat, Y, self.create_param);
        
        # loss+=tf.losses.get_regularization_loss;
        
        lr = tf.train.exponential_decay(NcfTraParm.learn_rate, global_step,
                                NcfTraParm.lr_decy_step,
                                NcfTraParm.lr_decy_rate,
                                staircase=True);
                                
        train_step = tf.train.AdagradOptimizer(lr). \
                    minimize(loss, global_step );
        
        
        save = tf.train.Saver();
        with tf.Session() as sess:
            if NcfTraParm.load_cache_rec:
                save.restore(sess,NcfTraParm.cache_rec_path);
            else:
                sess.run(tf.global_variables_initializer()); 
            
            now = time.time();
            for ep in range(NcfTraParm.epoch):
                sess.run(train_init_op);
                while True:
                    try:
                        _,vloss,gs=sess.run((train_step,loss,global_step));
                        if gs%(500) == 0:
                            print('ep%d\t loopstep:%d\t time:%.2f\t loss:%f'%(ep,gs,time.time()-now,vloss))
                            now=time.time();
                    except tf.errors.OutOfRangeError:
                        print('to end');
                        break  
                sess.run(test_init_op);
                vmae,vrmse,vloss=sess.run((tmae,trmse,loss)); 
                print('ep%d结束 \t eponloss=%f\t test_mae=%f test_rmse=%f\n'%(ep ,vloss,vmae,vrmse));
            
            if NcfTraParm.cache_rec_path != '':
                save.save(sess,NcfTraParm.cache_rec_path)
        pass;
    
    pass;




















if __name__ == '__main__':
    pass