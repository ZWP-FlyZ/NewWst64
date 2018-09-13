# -*- coding: utf-8 -*-
'''
Created on 2018年9月12日

@author: zwp12
'''

'''
ncf的改进性

'''

import tensorflow as tf;
import time;
from tools.ncf_tools import reoge_data,reoge_data3D;
from tools.fwrite import fwrite_append;
from tools.tfboard_sum import var_summaries;
import numpy as np;



class simple_ncf():
    '''
    >单ncf模型
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
#         mPu= tf.layers.dense(inputs=U,units=hid_f,
#                            #  kernel_initializer=initializers.random_normal(stddev=2),
#                             kernel_regularizer=reg_func);
#         mQs= tf.layers.dense(inputs=S,units=hid_f,
#                             # kernel_initializer=initializers.random_normal(stddev=2),
#                             kernel_regularizer=reg_func);
#         # 传统矩阵分解                    
#         mout= mPu * mQs;

        out = tf.concat([Pu,Qs],axis=1);        
        
        for unit in hid_units:
            out=tf.layers.dense(inputs=out,units=unit,
                                activation=hid_actfunc,
                                kernel_regularizer=reg_func);
            out=tf.layers.dropout(out,NcfCreParam.drop_p);                    
        
#         # 双模型混合                        
#         out =  tf.concat([mout,out],axis=1);
        print(out)
        # 输出层                       
        out=tf.layers.dense(inputs=out,units=1,
                            activation=hid_actfunc,
                            kernel_regularizer=reg_func);
        
        Py=out;                    
        # 误差                   
        loss = tf.reduce_mean(tf.losses.huber_loss(Y,out));
        tf.summary.scalar('loss',loss);
        # 评测误差
        mae = tf.reduce_mean(tf.abs(Y-out));
        tf.summary.scalar('mae',mae);
        rmse = tf.sqrt(tf.reduce_mean((Y-out)**2));
        tf.summary.scalar('rmse',rmse);
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
        
        loss+=tf.losses.get_regularization_loss();
        
        lr = tf.train.exponential_decay(NcfTraParm.learn_rate, global_step,
                                NcfTraParm.lr_decy_step,
                                NcfTraParm.lr_decy_rate,
                                staircase=True);
                                
        train_step = tf.train.AdagradOptimizer(lr). \
                    minimize(loss, global_step );
        
        summ_meg = tf.summary.merge_all();
        save = tf.train.Saver();
        with tf.Session() as sess:
            train_summ = tf.summary.FileWriter(NcfTraParm.summary_path+'/train',sess.graph);
            test_summ =tf.summary.FileWriter(NcfTraParm.summary_path+'/test');
            if NcfTraParm.load_cache_rec:
                save.restore(sess,NcfTraParm.cache_rec_path);
            else:
                sess.run(tf.global_variables_initializer()); 
            
            now = time.time();
            eptime = now;
            for ep in range(NcfTraParm.epoch):
                sess.run(train_init_op);
                while True:
                    try:
                        _,vloss,gs=sess.run((train_step,loss,global_step));
                        if gs%(500) == 0:
                            print('ep%d\t loopstep:%d\t time:%.2f\t loss:%f'%(ep,gs,time.time()-now,vloss))
                            summ = sess.run((summ_meg));
                            train_summ.add_summary(summ, gs);
                            now=time.time();
                    except tf.errors.OutOfRangeError:
                        break  
                sess.run(test_init_op);
                summ,vmae,vrmse,vloss=sess.run((summ_meg,tmae,trmse,loss));
                test_summ.add_summary(summ, ep);
                eps = '==================================================\n'
                eps += 'ep%d结束 \t eptime=%.2f\n' %(ep ,time.time()-eptime);
                eps += 'test_mae=%f test_rmse=%f\n'%(vmae,vrmse);
                eps += 'acttime=%s\n'%(time.asctime());
                eps += '==================================================\n'
                eptime = time.time();
                print(eps);
                if NcfTraParm.result_file_path != '': 
                    fwrite_append(NcfTraParm.result_file_path,eps);                
                 
                print('ep%d结束 \t eponloss=%f\t test_mae=%f test_rmse=%f\n'%(ep ,vloss,vmae,vrmse));
            
            if NcfTraParm.cache_rec_path != '':
                save.save(sess,NcfTraParm.cache_rec_path)
        pass;
    
    pass;

###########################################

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
        tf.summary.scalar('loss',loss);
        # 评测误差
        mae = tf.reduce_mean(tf.abs(Y-out));
        tf.summary.scalar('mae',mae);
        rmse = tf.sqrt(tf.reduce_mean((Y-out)**2));
        tf.summary.scalar('rmse',rmse);
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
        
        loss+=tf.losses.get_regularization_loss();
        
        lr = tf.train.exponential_decay(NcfTraParm.learn_rate, global_step,
                                NcfTraParm.lr_decy_step,
                                NcfTraParm.lr_decy_rate,
                                staircase=True);
                                
        train_step = tf.train.AdagradOptimizer(lr). \
                    minimize(loss, global_step );
        
        summ_meg = tf.summary.merge_all();
        save = tf.train.Saver();
        with tf.Session() as sess:
            train_summ = tf.summary.FileWriter(NcfTraParm.summary_path+'/train',sess.graph);
            test_summ =tf.summary.FileWriter(NcfTraParm.summary_path+'/test');
            if NcfTraParm.load_cache_rec:
                save.restore(sess,NcfTraParm.cache_rec_path);
            else:
                sess.run(tf.global_variables_initializer()); 
            
            now = time.time();
            eptime = now;
            for ep in range(NcfTraParm.epoch):
                sess.run(train_init_op);
                while True:
                    try:
                        _,vloss,gs=sess.run((train_step,loss,global_step));
                        if gs%(500) == 0:
                            print('ep%d\t loopstep:%d\t time:%.2f\t loss:%f'%(ep,gs,time.time()-now,vloss))
                            summ = sess.run((summ_meg));
                            train_summ.add_summary(summ, gs);
                            now=time.time();
                    except tf.errors.OutOfRangeError:
                        break  
                sess.run(test_init_op);
                summ,vmae,vrmse,vloss=sess.run((summ_meg,tmae,trmse,loss));
                test_summ.add_summary(summ, ep);
                eps = '==================================================\n'
                eps += 'ep%d结束 \t eptime=%.2f\n' %(ep ,time.time()-eptime);
                eps += 'test_mae=%f test_rmse=%f\n'%(vmae,vrmse);
                eps += 'acttime=%s\n'%(time.asctime());
                eps += '==================================================\n'
                eptime = time.time();
                print(eps);
                if NcfTraParm.result_file_path != '': 
                    fwrite_append(NcfTraParm.result_file_path,eps);                
                 
                print('ep%d结束 \t eponloss=%f\t test_mae=%f test_rmse=%f\n'%(ep ,vloss,vmae,vrmse));
            
            if NcfTraParm.cache_rec_path != '':
                save.save(sess,NcfTraParm.cache_rec_path)
        pass;
    
    pass;




###########################################


class ncf_baseline():
    '''
    >单ncf模型
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
        
        bu = tf.layers.dense(inputs=U,units=1,
                             kernel_initializer=tf.initializers.zeros,
                            kernel_regularizer=reg_func);
        
        bi = tf.layers.dense(inputs=S,units=1,
                            kernel_initializer=tf.initializers.zeros,
                            kernel_regularizer=reg_func);

        mu = tf.fill(tf.shape(bu),tf.cast(NcfCreParam.baseline_mu,tf.float32));
        # mu = tf.constant(NcfCreParam.baseline_mu,dtype=tf.float32,shape=tf.shape(bu));
        # baseline=tf.reshape(baseline,[-1,1]);

        out = tf.concat([Pu,Qs],axis=1);        
        
        for unit in hid_units:
            out=tf.layers.dense(inputs=out,units=unit,
                                activation=hid_actfunc,
                                kernel_regularizer=reg_func);
            out=tf.layers.dropout(out,NcfCreParam.drop_p);                    
        
        # 双模型混合                        
        out =  tf.concat([out,bu,bi,mu],axis=1);
        
        # 输出层                       
        out=tf.layers.dense(inputs=out,units=1,
                            activation=hid_actfunc,
                            kernel_regularizer=reg_func);
        Py=out;
        print(out)                    
        # 误差                   
        loss = tf.reduce_mean(tf.losses.huber_loss(Y,out));
        loss+=tf.losses.get_regularization_loss();
        tf.summary.scalar('loss',loss);
        # 评测误差
        mae = tf.reduce_mean(tf.abs(Y-out));
        tf.summary.scalar('mae',mae);
        rmse = tf.sqrt(tf.reduce_mean((Y-out)**2));
        tf.summary.scalar('rmse',rmse);
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
        
        loss+=tf.losses.get_regularization_loss();
        
        lr = tf.train.exponential_decay(NcfTraParm.learn_rate, global_step,
                                NcfTraParm.lr_decy_step,
                                NcfTraParm.lr_decy_rate,
                                staircase=True);
                                
        train_step = tf.train.AdagradOptimizer(lr). \
                    minimize(loss, global_step );
        
        summ_meg = tf.summary.merge_all();
        save = tf.train.Saver();
        with tf.Session() as sess:
            train_summ = tf.summary.FileWriter(NcfTraParm.summary_path+'/train',sess.graph);
            test_summ =tf.summary.FileWriter(NcfTraParm.summary_path+'/test');
            if NcfTraParm.load_cache_rec:
                save.restore(sess,NcfTraParm.cache_rec_path);
            else:
                sess.run(tf.global_variables_initializer()); 
            
            now = time.time();
            eptime = now;
            for ep in range(NcfTraParm.epoch):
                sess.run(train_init_op);
                while True:
                    try:
                        _,vloss,gs=sess.run((train_step,loss,global_step));
                        if gs%(500) == 0:
                            print('ep%d\t loopstep:%d\t time:%.2f\t loss:%f'%(ep,gs,time.time()-now,vloss))
                            summ = sess.run((summ_meg));
                            train_summ.add_summary(summ, gs);
                            now=time.time();
                    except tf.errors.OutOfRangeError:
                        break  
                sess.run(test_init_op);
                summ,vmae,vrmse,vloss=sess.run((summ_meg,tmae,trmse,loss));
                test_summ.add_summary(summ, ep);
                eps = '==================================================\n'
                eps += 'ep%d结束 \t eptime=%.2f\n' %(ep ,time.time()-eptime);
                eps += 'test_mae=%f test_rmse=%f\n'%(vmae,vrmse);
                eps += 'acttime=%s\n'%(time.asctime());
                eps += '==================================================\n'
                eptime = time.time();
                print(eps);
                if NcfTraParm.result_file_path != '': 
                    fwrite_append(NcfTraParm.result_file_path,eps);                
                 
                print('ep%d结束 \t eponloss=%f\t test_mae=%f test_rmse=%f\n'%(ep ,vloss,vmae,vrmse));
            
            if NcfTraParm.cache_rec_path != '':
                save.save(sess,NcfTraParm.cache_rec_path)
        pass;
    
    pass;

###########################################


class ncf_pp():
    '''
    >单ncf模型
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
#         self.label   = \
#             tf.placeholder(tf.float32, [None,self.sNum], 'label');
        pass;
    
    
        
    def toOneHot(self,feature):
        U = tf.one_hot(feature[:,0], self.uNum)
        S = tf.one_hot(feature[:,1],self.sNum);
        return U,S;
    
    def create_model(self,feat,Y,NcfCreParam,us_invked):
        hid_f = NcfCreParam.hid_feat;                          
        hid_units = NcfCreParam.hid_units;
        
        hid_actfunc =tf.nn.relu;
        # out_actfunc = tf.sigmoid;

        reg_func = tf.contrib.layers.l2_regularizer(NcfCreParam.reg_p);
        
        
        # ################# 处理 u_invked ###################### #
        Yj = tf.get_variable('Yj', [self.sNum,hid_f], 
                            initializer=tf.initializers.zeros, 
                            regularizer=reg_func);
        
        us_invked = tf.constant(us_invked,tf.float32);
        
        idx = tf.reshape(tf.cast(feat[:,0],tf.int32),[-1,1]);
        
        uivk = tf.gather_nd(us_invked,idx);
        
        Zj = tf.matmul(uivk,Yj);

        U,S = self.toOneHot(feat);
        
        # 初始化 隐含特征矩阵
        Pu = tf.layers.dense(inputs=U,units=hid_f,
                           #  kernel_initializer=initializers.random_normal(stddev=2),
                            kernel_regularizer=reg_func);
        Qs = tf.layers.dense(inputs=S,units=hid_f,
                             # kernel_initializer=initializers.random_normal(stddev=2),
                            kernel_regularizer=reg_func);
        
        bu = tf.layers.dense(inputs=U,units=1,
                             kernel_initializer=tf.initializers.zeros,
                            kernel_regularizer=reg_func);
        
        bi = tf.layers.dense(inputs=S,units=1,
                            kernel_initializer=tf.initializers.zeros,
                            kernel_regularizer=reg_func);

        mu = tf.fill(tf.shape(bu),tf.cast(NcfCreParam.baseline_mu,tf.float32));
        # mu = tf.constant(NcfCreParam.baseline_mu,dtype=tf.float32,shape=tf.shape(bu));
        # baseline=tf.reshape(baseline,[-1,1]);
        Zu = Pu+Zj;

        out = tf.concat([Zu,Qs],axis=1);        
        
        for unit in hid_units:
            out=tf.layers.dense(inputs=out,units=unit,
                                activation=hid_actfunc,
                                kernel_regularizer=reg_func);
            out=tf.layers.dropout(out,NcfCreParam.drop_p);                    
        
        # 双模型混合                        
        out =  tf.concat([out,bu,bi,mu],axis=1);
        
        # 输出层                       
        out=tf.layers.dense(inputs=out,units=1,
                            activation=hid_actfunc,
                            kernel_regularizer=reg_func);
        Py=out;
        print(out)                    
        # 误差                   
        loss = tf.reduce_mean(tf.losses.huber_loss(Y,out));
        loss+=tf.losses.get_regularization_loss();
        tf.summary.scalar('loss',loss);
        # 评测误差
        mae = tf.reduce_mean(tf.abs(Y-out));
        tf.summary.scalar('mae',mae);
        rmse = tf.sqrt(tf.reduce_mean((Y-out)**2));
        tf.summary.scalar('rmse',rmse);
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
            
        Py,loss,tmae,trmse= self.create_model(feat, Y, self.create_param,NcfTraParm.us_invked);
        
        loss+=tf.losses.get_regularization_loss();
        
        lr = tf.train.exponential_decay(NcfTraParm.learn_rate, global_step,
                                NcfTraParm.lr_decy_step,
                                NcfTraParm.lr_decy_rate,
                                staircase=True);
                                
        train_step = tf.train.AdagradOptimizer(lr). \
                    minimize(loss, global_step );
        
        summ_meg = tf.summary.merge_all();
        save = tf.train.Saver();
        with tf.Session() as sess:
            train_summ = tf.summary.FileWriter(NcfTraParm.summary_path+'/train',sess.graph);
            test_summ =tf.summary.FileWriter(NcfTraParm.summary_path+'/test');
            if NcfTraParm.load_cache_rec:
                save.restore(sess,NcfTraParm.cache_rec_path);
            else:
                sess.run(tf.global_variables_initializer()); 
            vvv = [[0]*(self.sNum+1)];
            now = time.time();
            eptime = now;
            for ep in range(NcfTraParm.epoch):
                sess.run(train_init_op);
                while True:
                    try:
                        _,vloss,gs=sess.run((train_step,loss,global_step));
                        if gs%(500) == 0:
                            print('ep%d\t loopstep:%d\t time:%.2f\t loss:%f'%(ep,gs,time.time()-now,vloss))
                            summ = sess.run((summ_meg));
                            train_summ.add_summary(summ, gs);
                            now=time.time();
                    except tf.errors.OutOfRangeError:
                        break  
                sess.run(test_init_op);
                summ,vmae,vrmse,vloss=sess.run((summ_meg,tmae,trmse,loss));
                test_summ.add_summary(summ, ep);
                eps = '==================================================\n'
                eps += 'ep%d结束 \t eptime=%.2f\n' %(ep ,time.time()-eptime);
                eps += 'test_mae=%f test_rmse=%f\n'%(vmae,vrmse);
                eps += 'acttime=%s\n'%(time.asctime());
                eps += '==================================================\n'
                eptime = time.time();
                print(eps);
                if NcfTraParm.result_file_path != '': 
                    fwrite_append(NcfTraParm.result_file_path,eps);                
                 
                print('ep%d结束 \t eponloss=%f\t test_mae=%f test_rmse=%f\n'%(ep ,vloss,vmae,vrmse));
            
            if NcfTraParm.cache_rec_path != '':
                save.save(sess,NcfTraParm.cache_rec_path)
        pass;
    
    pass;

###########################################

class ncf_pp_test():
    '''
    >单ncf模型
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
        self.u_invk_init = \
            tf.placeholder(tf.float32, [None,self.sNum+1]);
#         self.label   = \
#             tf.placeholder(tf.float32, [None,self.sNum], 'label');
        pass;
    
    
    def u_invk_op(self,u_invk,u_invk_cot):
        ucots  = np.array(u_invk_cot);
        mx = np.max(ucots);
        for ul in u_invk:
            if len(ul)<mx: 
                ul.extend([-1]*(mx-len(ul)));
        uls = np.array(u_invk)+1;
        ucots = (1.0/np.sqrt(ucots)).astype(np.float32);
        return uls,ucots;
        
    def toOneHot(self,feature):
        U = tf.one_hot(feature[:,0], self.uNum)
        S = tf.one_hot(feature[:,1],self.sNum);
        return U,S;
    
    def create_model(self,feat,Y,NcfCreParam,NcfTraParm):
        hid_f = NcfCreParam.hid_feat;                          
        hid_units = NcfCreParam.hid_units;
        
        hid_actfunc =tf.nn.relu;
        # out_actfunc = tf.sigmoid;

        reg_func = tf.contrib.layers.l2_regularizer(NcfCreParam.reg_p);
        
        
        # ################# 处理 u_invked ###################### #
        
        u_invked,u_invked_cot = self.u_invk_op(NcfTraParm.u_invked,
                                                NcfTraParm.u_invked_cot);
    
        def cond(i,_):
            return i<NcfTraParm.batchsize;

        def body(i,res):
            idx = tf.cast(feat[i,0],tf.int32);
            print(idx);
            hot = tf.one_hot(tf.gather(u_invked,idx)
                             ,self.sNum+1);
            hot = tf.reduce_sum(hot,axis=0,keepdims=True)*tf.gather(u_invked_cot,idx);
            
            res = tf.cond(tf.equal(i,tf.constant(0)),
                          lambda:hot,
                          lambda:tf.concat([res,hot],axis=0));
            return tf.add(i,1),res;
        
        _,uivk = tf.while_loop(cond,body,[tf.constant(0),self.u_invk_init]);
        uivk = tf.slice(uivk,[0,1],[NcfTraParm.batchsize,self.sNum]);
        
        Yj = tf.get_variable('Yj', [self.sNum,hid_f], 
                            initializer=tf.initializers.zeros, 
                            regularizer=reg_func);
        
        Zj = tf.matmul(uivk,Yj);

        U,S = self.toOneHot(feat);
        
        # 初始化 隐含特征矩阵
        Pu = tf.layers.dense(inputs=U,units=hid_f,
                           #  kernel_initializer=initializers.random_normal(stddev=2),
                            kernel_regularizer=reg_func);
        Qs = tf.layers.dense(inputs=S,units=hid_f,
                             # kernel_initializer=initializers.random_normal(stddev=2),
                            kernel_regularizer=reg_func);
        
        bu = tf.layers.dense(inputs=U,units=1,
                             kernel_initializer=tf.initializers.zeros,
                            kernel_regularizer=reg_func);
        
        bi = tf.layers.dense(inputs=S,units=1,
                            kernel_initializer=tf.initializers.zeros,
                            kernel_regularizer=reg_func);

        mu = tf.fill(tf.shape(bu),tf.cast(NcfCreParam.baseline_mu,tf.float32));
        # mu = tf.constant(NcfCreParam.baseline_mu,dtype=tf.float32,shape=tf.shape(bu));
        # baseline=tf.reshape(baseline,[-1,1]);
        Zu = Pu+Zj;

        out = tf.concat([Zu,Qs],axis=1);        
        
        for unit in hid_units:
            out=tf.layers.dense(inputs=out,units=unit,
                                activation=hid_actfunc,
                                kernel_regularizer=reg_func);
            out=tf.layers.dropout(out,NcfCreParam.drop_p);                    
        
        # 双模型混合                        
        out =  tf.concat([out,bu,bi,mu],axis=1);
        
        # 输出层                       
        out=tf.layers.dense(inputs=out,units=1,
                            activation=hid_actfunc,
                            kernel_regularizer=reg_func);
        Py=out;
        print(out)                    
        # 误差                   
        loss = tf.reduce_mean(tf.losses.huber_loss(Y,out));
        loss+=tf.losses.get_regularization_loss();
        tf.summary.scalar('loss',loss);
        # 评测误差
        mae = tf.reduce_mean(tf.abs(Y-out));
        tf.summary.scalar('mae',mae);
        rmse = tf.sqrt(tf.reduce_mean((Y-out)**2));
        tf.summary.scalar('rmse',rmse);
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
            
        Py,loss,tmae,trmse= self.create_model(feat, Y, self.create_param,NcfTraParm);
        
        loss+=tf.losses.get_regularization_loss();
        
        lr = tf.train.exponential_decay(NcfTraParm.learn_rate, global_step,
                                NcfTraParm.lr_decy_step,
                                NcfTraParm.lr_decy_rate,
                                staircase=True);
                                
        train_step = tf.train.AdagradOptimizer(lr). \
                    minimize(loss, global_step );
        
        summ_meg = tf.summary.merge_all();
        save = tf.train.Saver();
        with tf.Session() as sess:
            train_summ = tf.summary.FileWriter(NcfTraParm.summary_path+'/train',sess.graph);
            test_summ =tf.summary.FileWriter(NcfTraParm.summary_path+'/test');
            if NcfTraParm.load_cache_rec:
                save.restore(sess,NcfTraParm.cache_rec_path);
            else:
                sess.run(tf.global_variables_initializer()); 
            vvv = [[0]*(self.sNum+1)];
            now = time.time();
            eptime = now;
            for ep in range(NcfTraParm.epoch):
                sess.run(train_init_op);
                while True:
                    try:
                        _,vloss,gs=sess.run((train_step,loss,global_step),{self.u_invk_init:vvv});
                        if gs%(500) == 0:
                            print('ep%d\t loopstep:%d\t time:%.2f\t loss:%f'%(ep,gs,time.time()-now,vloss))
                            summ = sess.run((summ_meg),{self.u_invk_init:vvv});
                            train_summ.add_summary(summ, gs);
                            now=time.time();
                    except tf.errors.OutOfRangeError:
                        break  
                sess.run(test_init_op);
                summ,vmae,vrmse,vloss=sess.run((summ_meg,tmae,trmse,loss),{self.u_invk_init:vvv});
                test_summ.add_summary(summ, ep);
                eps = '==================================================\n'
                eps += 'ep%d结束 \t eptime=%.2f\n' %(ep ,time.time()-eptime);
                eps += 'test_mae=%f test_rmse=%f\n'%(vmae,vrmse);
                eps += 'acttime=%s\n'%(time.asctime());
                eps += '==================================================\n'
                eptime = time.time();
                print(eps);
                if NcfTraParm.result_file_path != '': 
                    fwrite_append(NcfTraParm.result_file_path,eps);                
                 
                print('ep%d结束 \t eponloss=%f\t test_mae=%f test_rmse=%f\n'%(ep ,vloss,vmae,vrmse));
            
            if NcfTraParm.cache_rec_path != '':
                save.save(sess,NcfTraParm.cache_rec_path)
        pass;
    
    pass;

###########################################



if __name__ == '__main__':
    pass