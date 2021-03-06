# -*- coding: utf-8 -*-
'''
Created on 2018年8月10日

@author: zwp
'''

import tensorflow as tf;
import time;
from tools.ncf_tools import reoge_data,reoge_data3D;
from tools.fwrite import fwrite_append;
from tools.tfboard_sum import var_summaries;

class hyb_ncf3D():
    '''
    >混合模型
    '''
    
    # 创建参数
    create_param = None;
    
    # 预测结果
    Py = None;
    # 损失
    loss = None;
    
    # 评测结果
    mae = None;
    rmse= None;
    

    
    def __init__(self,NcfCreParam3D):
        self.uNum,self.sNum,self.tNum=NcfCreParam3D.ust_shape;
        self.create_param = NcfCreParam3D;
#         self.feature = \
#             tf.placeholder(tf.float32, [None,self.uNum], 'feature');
#         self.label   = \
#             tf.placeholder(tf.float32, [None,self.sNum], 'label');
        pass;
    
    
    def _toOne(self,TG,O1,O2):
        TG[O1[0],O2[0]]=1;
    
    def toOneHot(self,feature):
        '''
        feature = [[U,S,T]]
        ''' 
        U = tf.one_hot(feature[:,0], self.uNum);
        S = tf.one_hot(feature[:,1],self.sNum);
        T = tf.one_hot(feature[:,2],self.tNum);
        U = tf.reshape(U,[-1,self.uNum,1]);
        S = tf.reshape(S,[-1,self.sNum,1]);
        T = tf.reshape(T,[-1,1,self.tNum]);
        UT = tf.reshape(tf.matmul(U,T),[-1,1,self.uNum,self.tNum]);
        ST = tf.reshape(tf.matmul(S,T),[-1,1,self.sNum,self.tNum]);
        return UT,ST;
        
    def chooser(self,W,inputs):
        c = W * inputs;
        print(c,W,inputs);
        c= tf.reduce_sum(c,axis=(2,3));
        return c;
        
        
    def create_model(self,feat,Y,NcfCreParam3D):
        hid_f = NcfCreParam3D.hid_feat;                          
        hid_units = NcfCreParam3D.hid_units;
        
        hid_actfunc =tf.nn.relu;
        # out_actfunc = tf.sigmoid;

        reg_func = tf.contrib.layers.l2_regularizer(NcfCreParam3D.reg_p);
         

        UT,ST = self.toOneHot(feat);
        
        PT = tf.get_variable('PT',(hid_f,self.uNum,self.tNum),
                             dtype=tf.float32,
                             initializer=tf.initializers.random_normal(stddev=1.0),
                             regularizer=reg_func
                            );
        QT = tf.get_variable('QT',(hid_f,self.sNum,self.tNum),
                             dtype=tf.float32,
                             initializer=tf.initializers.random_normal(stddev=1.0),
                             regularizer=reg_func
                            );
        mPT = tf.get_variable('mPT',(hid_f,self.uNum,self.tNum),
                             dtype=tf.float32,
                             initializer=tf.initializers.random_normal(stddev=1.0),
                             regularizer=reg_func
                            );
        mQT = tf.get_variable('mQT',(hid_f,self.sNum,self.tNum),
                             dtype=tf.float32,
                             initializer=tf.initializers.random_normal(stddev=1.0),
                             regularizer=reg_func
                            );
        # 初始化 隐含特征矩阵
        PTu = self.chooser(PT, UT);
        QTs = self.chooser(QT, ST);

        mPTu= self.chooser(mPT, UT);
        mQTs= self.chooser(mQT, ST);
        
        # 传统矩阵分解                    
        mout= mPTu * mQTs;

        out = tf.concat([PTu,QTs],axis=1);        
        
        for unit in hid_units:
            out=tf.layers.dense(inputs=out,units=unit,
                                activation=hid_actfunc,
                                kernel_regularizer=reg_func);
            out=tf.layers.dropout(out,NcfCreParam3D.drop_p);                    
        
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
    
    
    def train(self,NcfTraParm3D):

        
        train_data = reoge_data3D(NcfTraParm3D.train_data);
        test_data = reoge_data3D(NcfTraParm3D.test_data);
        testn = len(test_data[0]);
        global_step = tf.Variable(0,trainable=False,name='gs');
        ds = tf.data. \
                Dataset.from_tensor_slices(train_data);
        ds = ds.shuffle(1000).batch(NcfTraParm3D.batch_size);
        
        test_ds = tf.data.Dataset.from_tensor_slices(test_data);
        test_ds = test_ds.batch(testn);
        it = tf.data.Iterator.from_structure(ds.output_types,
                                            ds.output_shapes);
        
        feat,Y = it.get_next(); 
        train_init_op = it.make_initializer(ds);
        test_init_op = it.make_initializer(test_ds);   
            
        Py,loss,tmae,trmse= self.create_model(feat, Y, self.create_param);
        
        # loss+=tf.losses.get_regularization_loss;
        
        lr = tf.train.exponential_decay(NcfTraParm3D.learn_rate, global_step,
                                NcfTraParm3D.lr_decy_step,
                                NcfTraParm3D.lr_decy_rate,
                                staircase=True);
                                
        train_step = tf.train.AdagradOptimizer(lr). \
                    minimize(loss, global_step );
        
        
        save = tf.train.Saver();
        with tf.Session() as sess:
            with tf.device('/gpu:0'):
                if NcfTraParm3D.load_cache_rec:
                    save.restore(sess,NcfTraParm3D.cache_rec_path);
                else:
                    sess.run(tf.global_variables_initializer()); 
                
                now = time.time();
                for ep in range(NcfTraParm3D.epoch):
                    sess.run(train_init_op);
                    while True:
                        try:
                            _,vloss,gs=sess.run((train_step,loss,global_step));
                            if gs%(10) == 0:
                                print('ep%d\t loopstep:%d\t time:%.2f\t loss:%f'%(ep,gs,time.time()-now,vloss))
                                now=time.time();
                        except tf.errors.OutOfRangeError:
                            break  
                    sess.run(test_init_op);
                    vmae,vrmse,vloss=sess.run((tmae,trmse,loss)); 
                    print('ep%d结束 \t eponloss=%f\t test_mae=%f test_rmse=%f\n'%(ep ,vloss,vmae,vrmse));
                
                if NcfTraParm3D.cache_rec_path != '':
                    save.save(sess,NcfTraParm3D.cache_rec_path)
        pass;
    
    
    
    pass;


class hyb_ncf3D_test():
    '''
    >混合模型
    '''
    
    # 创建参数
    create_param = None;
    
    # 预测结果
    Py = None;
    # 损失
    loss = None;
    
    # 评测结果
    mae = None;
    rmse= None;
    

    
    def __init__(self,NcfCreParam3D):
        self.uNum,self.sNum,self.tNum=NcfCreParam3D.ust_shape;
        self.create_param = NcfCreParam3D;
#         self.feature = \
#             tf.placeholder(tf.float32, [None,self.uNum], 'feature');
#         self.label   = \
#             tf.placeholder(tf.float32, [None,self.sNum], 'label');
        pass;
    
    
  
    def toUSToneHot(self,feature):
        U = tf.one_hot(feature[:,0], self.uNum)
        S = tf.one_hot(feature[:,1],self.sNum);
        T = tf.one_hot(feature[:,2],self.tNum);
        
    def chooser(self,W,inputs):
        # W shape = [UorS,T,F]
        # input = [None,2];
        
        return tf.gather_nd(W,inputs);
        
        
    def create_model(self,feat,Y,NcfCreParam3D):
        hid_f = NcfCreParam3D.hid_feat;                          
        hid_units = NcfCreParam3D.hid_units;
        
        hid_actfunc =tf.nn.relu;
        # out_actfunc = tf.sigmoid;

        reg_func = tf.contrib.layers.l2_regularizer(NcfCreParam3D.reg_p);
         

        UT,ST = self.toUSToneHot(feat);
        
        PT = tf.get_variable('PT',(self.uNum,self.tNum,hid_f),
                             dtype=tf.float32,
                            
                             # initializer=tf.initializers.truncated_normal(stddev=0.1),
                             regularizer=reg_func
                            );
        QT = tf.get_variable('QT',(self.sNum,self.tNum,hid_f),
                             dtype=tf.float32,
                             # initializer=tf.initializers.truncated_normal(stddev=0.1),
                             regularizer=reg_func
                            );
        mPT = tf.get_variable('mPT',(self.uNum,self.tNum,hid_f),
                             dtype=tf.float32,
                             # initializer=tf.initializers.truncated_normal(stddev=0.1),
                             regularizer=reg_func
                            );
        mQT = tf.get_variable('mQT',(self.sNum,self.tNum,hid_f),
                             dtype=tf.float32,
                             # initializer=tf.initializers.truncated_normal(stddev=0.1),
                             regularizer=reg_func
                            );
        var_summaries(PT);
        var_summaries(QT);
        var_summaries(mPT);
        var_summaries(mQT);        
        # 初始化 隐含特征矩阵
        PTu = self.chooser(PT, UT);
        QTs = self.chooser(QT, ST);

        mPTu= self.chooser(mPT, UT);
        mQTs= self.chooser(mQT, ST);
        
        # 传统矩阵分解                    
        mout= mPTu * mQTs;

        out = tf.concat([PTu,QTs],axis=1);        
        
        for unit in hid_units:
            out=tf.layers.dense(inputs=out,units=unit,
                                activation=hid_actfunc,
                                kernel_regularizer=reg_func);
            out=tf.layers.dropout(out,NcfCreParam3D.drop_p);                    
        
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
        loss+=tf.losses.get_regularization_loss();
        # 评测误差
        mae = tf.reduce_mean(tf.abs(Y-out));
        rmse = tf.sqrt(tf.reduce_mean((Y-out)**2));
        return Py,loss,mae,rmse; 
    ############################# end  ##################################    
    
    
    def train(self,NcfTraParm3D):

        
        train_data = reoge_data3D(NcfTraParm3D.train_data);
        test_data = reoge_data3D(NcfTraParm3D.test_data);
        testn = len(test_data[0]);
        print(testn);
        global_step = tf.Variable(0,trainable=False,name='gs');
        ds = tf.data. \
                Dataset.from_tensor_slices(train_data);
        ds = ds.shuffle(100).batch(NcfTraParm3D.batch_size);
        
        test_ds = tf.data.Dataset.from_tensor_slices(test_data);
        test_ds = test_ds.batch(testn);
        it = tf.data.Iterator.from_structure(ds.output_types,
                                            ds.output_shapes);
        
        feat,Y = it.get_next(); 
        train_init_op = it.make_initializer(ds);
        test_init_op = it.make_initializer(test_ds);   
            
        Py,loss,tmae,trmse= self.create_model(feat, Y, self.create_param);
        tf.summary.scalar('loss',loss);
        tf.summary.scalar('mae',tmae);
        tf.summary.scalar('rmse',trmse);
        
        # loss+=tf.losses.get_regularization_loss;
        
        lr = tf.train.exponential_decay(NcfTraParm3D.learn_rate, global_step,
                                NcfTraParm3D.lr_decy_step,
                                NcfTraParm3D.lr_decy_rate,
                                staircase=True);
                                
        train_step = tf.train.AdagradOptimizer(lr). \
                    minimize(loss, global_step );
        summar_meg = tf.summary.merge_all();
        
        save = tf.train.Saver();
        with tf.Session() as sess:
            with tf.device('/cpu:0'):
                train_summ=tf.summary.FileWriter(NcfTraParm3D.summary_path+'/train',sess.graph);
                test_summ = tf.summary.FileWriter(NcfTraParm3D.summary_path+'/test',sess.graph);
                if NcfTraParm3D.load_cache_rec:
                    save.restore(sess,NcfTraParm3D.cache_rec_path);
                else:
                    sess.run(tf.global_variables_initializer()); 
                
                now = time.time();
                eptime = now;
                for ep in range(NcfTraParm3D.epoch):
                    sess.run(train_init_op);
                    while True:
                        try:
                            _,vloss,gs=sess.run((train_step,loss,global_step));
                            
                            if gs%(100) == 0:
                                print('ep%d\t loopstep:%d\t time:%.2f\t loss:%f'%(ep,gs,time.time()-now,vloss))
                                now=time.time();
                                summ = sess.run(summar_meg)
                                train_summ.add_summary(summ,gs);
                        except tf.errors.OutOfRangeError:
                            break  
                    sess.run(test_init_op);
                    summ,vmae,vrmse,vloss=sess.run((summar_meg,tmae,trmse,loss));
                    test_summ.add_summary(summ, gs);
                    eps = '==================================================\n'
                    eps += 'ep%d结束 \t eptime=%.2f\n' %(ep ,time.time()-eptime);
                    eps += 'test_mae=%f test_rmse=%f\n'%(vmae,vrmse);
                    eps += 'acttime=%s\n'%(time.asctime());
                    eps += '==================================================\n'
                    eptime = time.time();
                    print(eps);
                    if NcfTraParm3D.result_file_path != '': 
                        fwrite_append(NcfTraParm3D.result_file_path,eps);
                
                if NcfTraParm3D.cache_rec_path != '':
                    save.save(sess,NcfTraParm3D.cache_rec_path)
                train_summ.close();
                test_summ.close();
    pass;

class simple_ncf3D():
    '''
    >将时间数据分解在u,s,t三个特征矩阵上。
    '''
    
    # 创建参数
    create_param = None;
    
    # 预测结果
    Py = None;
    # 损失
    loss = None;
    
    # 评测结果
    mae = None;
    rmse= None;
    

    
    def __init__(self,NcfCreParam3D):
        self.uNum,self.sNum,self.tNum=NcfCreParam3D.ust_shape;
        self.create_param = NcfCreParam3D;
#         self.feature = \
#             tf.placeholder(tf.float32, [None,self.uNum], 'feature');
#         self.label   = \
#             tf.placeholder(tf.float32, [None,self.sNum], 'label');
        pass;
    
    
  
    def toUSToneHot(self,feature):
        U = tf.one_hot(feature[:,0], self.uNum)
        S = tf.one_hot(feature[:,1],self.sNum);
        T = tf.one_hot(feature[:,2],self.tNum);
        return U,S,T;

        
        
    def create_model(self,feat,Y,NcfCreParam3D):
        hid_f = NcfCreParam3D.hid_feat;                          
        hid_units = NcfCreParam3D.hid_units;
        
        hid_actfunc =tf.nn.relu;
        # out_actfunc = tf.sigmoid;

        reg_func = tf.contrib.layers.l2_regularizer(NcfCreParam3D.reg_p);
         
        U,S,T = self.toUSToneHot(feat);
        
        
        Pu = tf.layers.dense(inputs=U,units=hid_f,use_bias=False,
                           #  kernel_initializer=initializers.random_normal(stddev=2),
                            kernel_regularizer=reg_func);

        Qs = tf.layers.dense(inputs=S,units=hid_f,use_bias=False,
                           #  kernel_initializer=initializers.random_normal(stddev=2),
                            kernel_regularizer=reg_func);
        Rt = tf.layers.dense(inputs=T,units=hid_f,use_bias=False,
                           #  kernel_initializer=initializers.random_normal(stddev=2),
                            kernel_regularizer=reg_func);
        
#         P = tf.get_variable('P',(self.uNum,hid_f),
#                              dtype=tf.float32,
#                              initializer=tf.initializers.truncated_normal(stddev=0.1),
#                              regularizer=reg_func
#                             );
#         Q = tf.get_variable('Q',(self.sNum,hid_f),
#                              dtype=tf.float32,
#                              initializer=tf.initializers.truncated_normal(stddev=0.1),
#                              regularizer=reg_func
#                             );
#         R = tf.get_variable('R',(self.tNum,hid_f),
#                              dtype=tf.float32,
#                              initializer=tf.initializers.truncated_normal(stddev=0.1),
#                              regularizer=reg_func
#                             );

        
        out = tf.concat([Pu,Qs,Rt],axis=1);        
        
        for unit in hid_units:
            out=tf.layers.dense(inputs=out,units=unit,
                                activation=hid_actfunc,
                                kernel_regularizer=reg_func);
            out=tf.layers.dropout(out,NcfCreParam3D.drop_p);                    
        
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
        loss+=tf.losses.get_regularization_loss();
        # 评测误差
        mae = tf.reduce_mean(tf.abs(Y-out));
        rmse = tf.sqrt(tf.reduce_mean((Y-out)**2));
        return Py,loss,mae,rmse; 
    ############################# end  ##################################    
    
    
    def train(self,NcfTraParm3D):

        # 去除无效数据，分割特征和标签
        train_data = reoge_data3D(NcfTraParm3D.train_data);
        test_data = reoge_data3D(NcfTraParm3D.test_data);
        testn = len(test_data[0]);
        print(testn);
        global_step = tf.Variable(0,trainable=False,name='gs');
        ds = tf.data. \
                Dataset.from_tensor_slices(train_data);
        ds = ds.shuffle(100).batch(NcfTraParm3D.batch_size);
        
        test_ds = tf.data.Dataset.from_tensor_slices(test_data);
        test_ds = test_ds.batch(testn//20);
        it = tf.data.Iterator.from_structure(ds.output_types,
                                            ds.output_shapes);
        
        feat,Y = it.get_next(); 
        train_init_op = it.make_initializer(ds);
        test_init_op = it.make_initializer(test_ds);   
            
        Py,loss,tmae,trmse= self.create_model(feat, Y, self.create_param);
        tf.summary.scalar('loss',loss);
        tf.summary.scalar('mae',tmae);
        tf.summary.scalar('rmse',trmse);
        
        # loss+=tf.losses.get_regularization_loss;
        
        lr = tf.train.exponential_decay(NcfTraParm3D.learn_rate, global_step,
                                NcfTraParm3D.lr_decy_step,
                                NcfTraParm3D.lr_decy_rate,
                                staircase=True);
                                
        train_step = tf.train.AdagradOptimizer(lr). \
                    minimize(loss, global_step );
        summar_meg = tf.summary.merge_all();
        
        save = tf.train.Saver();
        with tf.Session() as sess:
            train_summ=tf.summary.FileWriter(NcfTraParm3D.summary_path+'/train',sess.graph);
            test_summ = tf.summary.FileWriter(NcfTraParm3D.summary_path+'/test',sess.graph);
            if NcfTraParm3D.load_cache_rec:
                save.restore(sess,NcfTraParm3D.cache_rec_path);
            else:
                sess.run(tf.global_variables_initializer()); 
            
            now = time.time();
            eptime = now;
            for ep in range(NcfTraParm3D.epoch):
                sess.run(train_init_op);
                while True:
                    try:
                        _,vloss,gs=sess.run((train_step,loss,global_step));
                        
                        if gs%(1000) == 0:
                            print('ep%d\t loopstep:%d\t time:%.2f\t loss:%f'%(ep,gs,time.time()-now,vloss))
                            now=time.time();
                            summ = sess.run(summar_meg)
                            train_summ.add_summary(summ,gs);
                    except tf.errors.OutOfRangeError:
                        break  
                sess.run(test_init_op);
                summae=0;
                while True:
                    try:
                        summ,vmae,vrmse,vloss=sess.run((summar_meg,tmae,trmse,loss));
                        summae+=vmae;
                    except tf.errors.OutOfRangeError:
                        break;
                summae/=20;
                eps = '==================================================\n'
                eps += 'ep%d结束 \t eptime=%.2f\n' %(ep ,time.time()-eptime);
                eps += 'test_mae=%f test_rmse=%f\n'%(summae,vrmse);
                eps += 'acttime=%s\n'%(time.asctime());
                eps += '==================================================\n'
                eptime = time.time();
                print(eps);
                if NcfTraParm3D.result_file_path != '': 
                    fwrite_append(NcfTraParm3D.result_file_path,eps);
            if NcfTraParm3D.cache_rec_path != '':
                save.save(sess,NcfTraParm3D.cache_rec_path)
            train_summ.close();
            test_summ.close();
    pass;

if __name__ == '__main__':
    pass