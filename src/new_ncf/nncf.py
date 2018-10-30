# -*- coding: utf-8 -*-
'''
Created on 2018年10月30日

@author: zwp12
'''

import tensorflow as tf;
import time;
from tools.ncf_tools import reoge_data,reoge_data3D;
from tools.fwrite import fwrite_append;
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


if __name__ == '__main__':
    pass