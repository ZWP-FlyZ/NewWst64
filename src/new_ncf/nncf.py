# -*- coding: utf-8 -*-
'''
Created on 2018年10月30日

@author: zwp12
'''

import tensorflow as tf;
import time;
from tools.ncf_tools import reoge_data,reoge_data3D;
from tools.fwrite import fwrite_append;
from tensorflow import keras;
from tensorflow.keras.layers import LSTM;
# from new_ncf.ncf_param import NcfCreParamUST,NcfTraParmUST;

class simple_ncfUST():
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
    

    
    def __init__(self,NcfCreParamUST):
        self.uNum,self.sNum,self.tNum=NcfCreParamUST.ust_shape;
        self.create_param = NcfCreParamUST;
        pass;
    
    def toUSToneHot(self,feature):
        U = tf.one_hot(feature[:,0], self.uNum)
        S = tf.one_hot(feature[:,1],self.sNum);
        T = tf.one_hot(feature[:,2],self.tNum);
        return U,S,T;


    
    def get_hid_kernel(self,name,shape=None,regularizer=None):
        
        # 获取不可重复的核参数
        with tf.variable_scope('hid_kernel',reuse=tf.AUTO_REUSE):
            v = tf.get_variable(name, shape,
                    dtype=tf.float32, 
#                     initializer=tf.initializers.truncated_normal, 
                    regularizer = regularizer)
        return v;
    
    def get_ncf_nn_front(self,Pu,Qs,Rt,hid_units,activates,regularizer=None):
        # 获取的ncf的前向传播网络
        hid_actf,out_actf = activates;
        out = tf.concat([Pu,Qs,Rt],axis=1); 
        for i,unit in enumerate(hid_units):
            name = 'front_layer%d_unit%d'%(i,unit);
            out = tf.layers.dense(out, unit, 
                                activation=hid_actf,
                                kernel_regularizer=regularizer,
                                name=name, reuse=tf.AUTO_REUSE);
        name = 'front_out';
        out=tf.layers.dense(inputs=out,units=1,
                            activation=out_actf,
                            kernel_regularizer=regularizer,
                            name=name, reuse=tf.AUTO_REUSE);    
        return out;
            
    
    def create_ncf_model(self,feat,Y,NcfCreParamUST):
        hid_f = NcfCreParamUST.hid_feat;                          
        hid_units = NcfCreParamUST.hid_units;
        
        hid_actfunc =tf.nn.relu;        
    
        reg_func = tf.keras.regularizers.l2(NcfCreParamUST.reg_p);
         
        U,S,T = self.toUSToneHot(feat); 
        
        P = self.get_hid_kernel('P',[self.uNum,hid_f], reg_func);
        Q = self.get_hid_kernel('Q',[self.sNum,hid_f], reg_func); 
        R = self.get_hid_kernel('R',[self.tNum,hid_f], reg_func);
        
        
        Pu = tf.matmul(U,P);
        Qs = tf.matmul(S,Q);
        Rt = tf.matmul(T,R);
        print(Pu,Qs,Rt);
        # 全连接层
        out = self.get_ncf_nn_front(Pu, Qs, Rt, hid_units,
                    (hid_actfunc,hid_actfunc), reg_func);
        print(out);
               
        Py=out;                    
        # 误差                   
        loss = tf.reduce_mean(tf.losses.huber_loss(Y,out));
        loss+=tf.losses.get_regularization_loss();
        # 评测误差
        mae = tf.reduce_mean(tf.abs(Y-out));
        rmse = tf.sqrt(tf.reduce_mean((Y-out)**2));
        return Py,loss,mae,rmse;         
    
    ############################# end  ##################################    
    
    
    def train_ncf(self,NcfTraParmUST):

        # 去除无效数据，分割特征和标签
        train_data = reoge_data3D(NcfTraParmUST.train_data);
        test_data = reoge_data3D(NcfTraParmUST.test_data);
        testn = len(test_data[0]);
        print(testn);
        global_step = tf.Variable(0,trainable=False,name='gs');
        ds = tf.data. \
                Dataset.from_tensor_slices(train_data);
        ds = ds.shuffle(1000).batch(NcfTraParmUST.batch_size);
        
        test_ds = tf.data.Dataset.from_tensor_slices(test_data);
        test_ds = test_ds.batch(testn//20);
        it = tf.data.Iterator.from_structure(ds.output_types,
                                            ds.output_shapes);
        
        feat,Y = it.get_next(); 
        train_init_op = it.make_initializer(ds);
        test_init_op = it.make_initializer(test_ds);   
            
        Py,loss,tmae,trmse= self.create_ncf_model(feat, Y, self.create_param);
        tf.summary.scalar('loss',loss);
        tf.summary.scalar('mae',tmae);
        tf.summary.scalar('rmse',trmse);
        
        # loss+=tf.losses.get_regularization_loss;
        
        lr = tf.train.exponential_decay(NcfTraParmUST.learn_rate, global_step,
                                NcfTraParmUST.lr_decy_step,
                                NcfTraParmUST.lr_decy_rate,
                                staircase=True);
                                
        train_step = tf.train.AdagradOptimizer(lr). \
                    minimize(loss, global_step );
        summar_meg = tf.summary.merge_all();
        
        save = tf.train.Saver();
        with tf.Session() as sess:
            train_summ=tf.summary.FileWriter(NcfTraParmUST.summary_path+'/train',sess.graph);
            test_summ = tf.summary.FileWriter(NcfTraParmUST.summary_path+'/test',sess.graph);
            if NcfTraParmUST.load_cache_rec:
                save.restore(sess,NcfTraParmUST.cache_rec_path);
            else:
                sess.run(tf.global_variables_initializer()); 
            
            now = time.time();
            eptime = now;
            for ep in range(NcfTraParmUST.epoch):
                sess.run(train_init_op);
                # 训练
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
                
                # 评估 
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
                if NcfTraParmUST.result_file_path != '': 
                    fwrite_append(NcfTraParmUST.result_file_path,eps);
            
            if NcfTraParmUST.cache_rec_path != '':
                save.save(sess,NcfTraParmUST.cache_rec_path)
            train_summ.close();
            test_summ.close();
    pass;

    def train_rnn(self,NcfTraParmUST):
        pass;
    

class gru_model(tf.keras.Model):
    def __init__(self, hid_size, units):
        super(gru_model, self).__init__()
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






    
class ncf_rnnUST():
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
    

    
    def __init__(self,NcfCreParamUST):
        self.uNum,self.sNum,self.tNum=NcfCreParamUST.ust_shape;
        self.create_param = NcfCreParamUST;
        pass;
    
    def toUSToneHot(self,feature):
        U = tf.one_hot(feature[:,0], self.uNum)
        S = tf.one_hot(feature[:,1],self.sNum);
        T = tf.one_hot(feature[:,2],self.tNum);
        return U,S,T;


    
    def get_hid_kernel(self,name,shape=None,regularizer=None):
        
        # 获取不可重复的核参数
        with tf.variable_scope('hid_kernel',reuse=tf.AUTO_REUSE):
            v = tf.get_variable(name, shape,
                    dtype=tf.float32, 
#                     initializer=tf.initializers.truncated_normal, 
                    regularizer = regularizer)
        return v;
    
    def get_ncf_nn_front(self,Pu,Qs,Rt,hid_units,activates,regularizer=None):
        # 获取的ncf的前向传播网络
        hid_actf,out_actf = activates;
        out = tf.concat([Pu,Qs,Rt],axis=1); 
        for i,unit in enumerate(hid_units):
            name = 'front_layer%d_unit%d'%(i,unit);
            out = tf.layers.dense(out, unit, 
                                activation=hid_actf,
                                kernel_regularizer=regularizer,
                                name=name, reuse=tf.AUTO_REUSE);
        name = 'front_out';
        out=tf.layers.dense(inputs=out,units=1,
                            activation=out_actf,
                            kernel_regularizer=regularizer,
                            name=name, reuse=tf.AUTO_REUSE);    
        return out;
            
    
    def create_ncf_model(self,feat,Y,NcfCreParamUST):
        hid_f = NcfCreParamUST.hid_feat;                          
        hid_units = NcfCreParamUST.hid_units;
        
        hid_actfunc =tf.nn.relu;        
    
        reg_func = tf.keras.regularizers.l2(NcfCreParamUST.reg_p);
         
        U,S,T = self.toUSToneHot(feat); 
        
        P = self.get_hid_kernel('P',[self.uNum,hid_f], reg_func);
        Q = self.get_hid_kernel('Q',[self.sNum,hid_f], reg_func); 
        R = self.get_hid_kernel('R',[self.tNum,hid_f], reg_func);
        
        
        Pu = tf.matmul(U,P);
        Qs = tf.matmul(S,Q);
        Rt = tf.matmul(T,R);
        print(Pu,Qs,Rt);
        # 全连接层
        out = self.get_ncf_nn_front(Pu, Qs, Rt, hid_units,
                    (hid_actfunc,hid_actfunc), reg_func);
        print(out);
               
        Py=out;                    
        # 误差                   
        loss = tf.reduce_mean(tf.losses.huber_loss(Y,out));
        loss+=tf.losses.get_regularization_loss();
        # 评测误差
        mae = tf.reduce_mean(tf.abs(Y-out));
        rmse = tf.sqrt(tf.reduce_mean((Y-out)**2));
        return Py,loss,mae,rmse;         
    
    ############################# end  ##################################    
    
    def create_rnn_model(self,x,y,NcfTraParmUST):
        hid_f = self.create_param.hid_feat;
        unit = NcfTraParmUST.rnn_unit;
        
        gru = gru_model(hid_f,unit);
        py = gru(x);
        
        loss = tf.reduce_mean(tf.losses.mean_squared_error(y,py));
        mae = tf.reduce_mean(tf.abs(y-py));
        
        return py,loss,mae,gru;

        

    def conbine_train(self,NcfTraParmUST):
        
        
        ################################ 定义 ncf ##################################
        # 去除无效数据，分割特征和标签
        train_data = reoge_data3D(NcfTraParmUST.train_data);
        test_data = reoge_data3D(NcfTraParmUST.test_data);
        testn = len(test_data[0]);
        print(testn);
        global_step = tf.Variable(0,trainable=False,name='gs');
        ds = tf.data. \
                Dataset.from_tensor_slices(train_data);
        ds = ds.shuffle(1000).batch(NcfTraParmUST.batch_size);
        
        test_ds = tf.data.Dataset.from_tensor_slices(test_data);
        test_ds = test_ds.batch(testn//20);
        it = tf.data.Iterator.from_structure(ds.output_types,
                                            ds.output_shapes);
        
        feat,Y = it.get_next(); 
        train_init_op = it.make_initializer(ds);
        test_init_op = it.make_initializer(test_ds);   
            
        Py,loss,tmae,trmse= self.create_ncf_model(feat, Y, self.create_param);
        tf.summary.scalar('loss',loss);
        tf.summary.scalar('mae',tmae);
        tf.summary.scalar('rmse',trmse);
        
        # loss+=tf.losses.get_regularization_loss;
        
        lr = tf.train.exponential_decay(NcfTraParmUST.learn_rate, global_step,
                                NcfTraParmUST.lr_decy_step,
                                NcfTraParmUST.lr_decy_rate,
                                staircase=True);
                                
        train_step = tf.train.AdagradOptimizer(lr). \
                    minimize(loss, global_step );
        summar_meg = tf.summary.merge_all();
        
        
        ##################################### 定义 rnn #####################################
        
        seq_len = NcfTraParmUST.seq_len;
        t_strat,t_end = NcfTraParmUST.time_range;
        
        rnn_lr = NcfTraParmUST.rnn_learn_rat;
        rnn_epoch = NcfTraParmUST.rnn_epoch;
        
        R = self.get_hid_kernel('R');
        nR = R[t_strat:t_end];
        splitedR = tf.reshape(nR,[-1,seq_len+1,R.shape[1]]);
        print(splitedR);
        rnnds = tf.data.Dataset.from_tensor_slices(splitedR);
    #     ds.map(lambda i:(i[:-1],i[1:]));
        rnnit = rnnds.make_initializable_iterator();
        nextitem = rnnit.get_next();
        x,y=tf.expand_dims(nextitem[:-1],0),\
                tf.expand_dims(nextitem[1:],0);
        
        x = tf.cast(x,tf.float32);        
                
        rnnpy,rnnloss,rnnmae,rnngru=self.create_rnn_model(x,y,NcfTraParmUST);
        
        
        ########################## 定义 rnn 预测 #########################
        # 处理需要预测时间片的时间特征
        splitedR = tf.reshape(R[t_end-1],[-1,1,R.shape[1]]);
        pRet = rnngru(splitedR);
        pRet = tf.reshape(pRet,[1,R.shape[1]]);
        newR = tf.concat([R[:t_end],pRet,R[t_end+1:]],axis=0);
        newP = self.get_hid_kernel('P');
        newQ = self.get_hid_kernel('Q');
        
        predict_data = reoge_data3D(NcfTraParmUST.ts_train_data);
        rnn_ds = tf.data.Dataset.from_tensor_slices(predict_data);
        rnn_ds = rnn_ds.batch(len(predict_data[0])//20);
        pre_it = rnn_ds.make_one_shot_iterator();
        perfeat,preY = pre_it.get_next();
        U,S,T = self.toUSToneHot(perfeat);
        newPu = tf.matmul(U,newP);
        newQs = tf.matmul(S,newQ);
        newRt = tf.matmul(T,newR);
        ppy = self.get_ncf_nn_front(newPu, newQs, newRt, 
                    self.create_param.hid_units,
                    (tf.nn.relu,tf.nn.relu), None);
        print(tf.trainable_variables())
        pre_mae = tf.reduce_mean(tf.abs(ppy-preY));
        Ret = R[t_end];
        

        
        train_op = tf.train.AdamOptimizer(rnn_lr).minimize(rnnloss);
        
        save = tf.train.Saver();
        with tf.Session() as sess:
            train_summ=tf.summary.FileWriter(NcfTraParmUST.summary_path+'/train',sess.graph);
            test_summ = tf.summary.FileWriter(NcfTraParmUST.summary_path+'/test',sess.graph);
            if NcfTraParmUST.load_cache_rec:
                save.restore(sess,NcfTraParmUST.cache_rec_path);
            else:
                sess.run(tf.global_variables_initializer()); 
                now = time.time();
                eptime = now;
                for ep in range(NcfTraParmUST.epoch):
                    sess.run(train_init_op);
                    # 训练
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
                    
                    # 评估 
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
                    if NcfTraParmUST.result_file_path != '': 
                        fwrite_append(NcfTraParmUST.result_file_path,eps);
                
                if NcfTraParmUST.cache_rec_path != '':
                    save.save(sess,NcfTraParmUST.cache_rec_path)
                train_summ.close();
                test_summ.close();  
            ############## end else ###################
            
        ########################## RNN ##########################
            
            for ep in range(rnn_epoch):
                sess.run(rnnit.initializer)
                maesum=0.0;
                cot=0;
                while True:
                    try:
                        _,vx,vy,vpy,vloss,vmae,vpret,urrr=sess.run([train_op,x,y,rnnpy,rnnloss,rnnmae,pRet,Ret])
                        maesum+=vmae;
                        cot+=1;
#                         print('x:\n',vx);
#                         print('y:\n',vy);
#                         print('py:\n',vpy);
#                         print('loss:\n',vloss);
#                         print('mae:\n',vmae);
#                         print('pRet:\n',vpret);
#                         print('------------------------')
                    except tf.errors.OutOfRangeError:break;
                maesum/=cot;
#                 print('pRet:\n',vpret,urrr);
                print('ep%d end mae=%f'%(ep,maesum));
            
            
            predict_sum_mae =0.0;
            predict_cot=0;
            while True:
                    try:
                        vmae=sess.run(pre_mae)
                        predict_sum_mae+=vmae;
                        predict_cot+=1;
                    except tf.errors.OutOfRangeError:break;    
            print('all predict mae',predict_sum_mae/predict_cot);
            
            
            
            
    
    def train_ncf(self,NcfTraParmUST):

        # 去除无效数据，分割特征和标签
        train_data = reoge_data3D(NcfTraParmUST.train_data);
        test_data = reoge_data3D(NcfTraParmUST.test_data);
        testn = len(test_data[0]);
        print(testn);
        global_step = tf.Variable(0,trainable=False,name='gs');
        ds = tf.data. \
                Dataset.from_tensor_slices(train_data);
        ds = ds.shuffle(1000).batch(NcfTraParmUST.batch_size);
        
        test_ds = tf.data.Dataset.from_tensor_slices(test_data);
        test_ds = test_ds.batch(testn//20);
        it = tf.data.Iterator.from_structure(ds.output_types,
                                            ds.output_shapes);
        
        feat,Y = it.get_next(); 
        train_init_op = it.make_initializer(ds);
        test_init_op = it.make_initializer(test_ds);   
            
        Py,loss,tmae,trmse= self.create_ncf_model(feat, Y, self.create_param);
        tf.summary.scalar('loss',loss);
        tf.summary.scalar('mae',tmae);
        tf.summary.scalar('rmse',trmse);
        
        # loss+=tf.losses.get_regularization_loss;
        
        lr = tf.train.exponential_decay(NcfTraParmUST.learn_rate, global_step,
                                NcfTraParmUST.lr_decy_step,
                                NcfTraParmUST.lr_decy_rate,
                                staircase=True);
                                
        train_step = tf.train.AdagradOptimizer(lr). \
                    minimize(loss, global_step );
        summar_meg = tf.summary.merge_all();
        
        save = tf.train.Saver();
        with tf.Session() as sess:
            train_summ=tf.summary.FileWriter(NcfTraParmUST.summary_path+'/train',sess.graph);
            test_summ = tf.summary.FileWriter(NcfTraParmUST.summary_path+'/test',sess.graph);
            if NcfTraParmUST.load_cache_rec:
                save.restore(sess,NcfTraParmUST.cache_rec_path);
            else:
                sess.run(tf.global_variables_initializer()); 
            
            now = time.time();
            eptime = now;
            for ep in range(NcfTraParmUST.epoch):
                sess.run(train_init_op);
                # 训练
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
                
                # 评估 
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
                if NcfTraParmUST.result_file_path != '': 
                    fwrite_append(NcfTraParmUST.result_file_path,eps);
            
            if NcfTraParmUST.cache_rec_path != '':
                save.save(sess,NcfTraParmUST.cache_rec_path)
            train_summ.close();
            test_summ.close();
    pass;

    def train_rnn(self,NcfTraParmUST):
        hid_f = self.create_param.hid_feat;
        unit = NcfTraParmUST.rnn_unit;
        seq_len = NcfTraParmUST.seq_len;
        t_strat,t_end = NcfTraParmUST.time_range;
        
        rnn_lr = NcfTraParmUST.rnn_learn_rat;
        rnn_epoch = NcfTraParmUST.rnn_epoch;
        gru = gru_model(hid_f,unit);
        
        R = self.get_hid_kernel('R');
        R = R[t_strat:t_end];
        splitedR = tf.reshape(R,[-1,seq_len+1,hid_f]);
        print(splitedR);
        ds = tf.data.Dataset.from_tensor_slices(splitedR);
    #     ds.map(lambda i:(i[:-1],i[1:]));
        it = ds.make_initializable_iterator();
        nextitem = it.get_next();
        x,y=tf.expand_dims(nextitem[:-1],0),\
                tf.expand_dims(nextitem[1:],0);
        
        x = tf.cast(x,tf.float32);        
        
        py = gru(x);
        
        loss = tf.reduce_mean(tf.losses.mean_squared_error(y,py));
        mae = tf.reduce_mean(tf.abs(y-py));
        
        train_op = tf.train.AdamOptimizer(rnn_lr).minimize(loss);
        save = tf.train.Saver();
        with tf.Session() as sess:
            if NcfTraParmUST.load_cache_rec:
                save.restore(sess,NcfTraParmUST.cache_rec_path);
            else:
                sess.run(tf.global_variables_initializer())
            for ep in range(rnn_epoch):
                sess.run(it.initializer)
                maesum=0.0;
                cot=0;
                while True:
                    try:
                        _,vx,vy,vpy,vloss,vmae=sess.run([train_op,x,y,py,loss,mae])
                        maesum+=vmae;
                        cot+=1;
#                         print('x:\n',vx);
#                         print('y:\n',vy);
#                         print('py:\n',vpy);
#                         print('loss:\n',vloss);
#                         print('mae:\n',vmae);
                        print('------------------------')
                    except tf.errors.OutOfRangeError:break;
                maesum/=cot;
                print('ep%d end mae=%f'%(ep,maesum));
                        
        pass;    
    
    
if __name__ == '__main__':
    pass