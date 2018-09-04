# -*- coding: utf-8 -*-
'''
Created on 2018年8月7日

@author: zwp12
'''

class NcfCreParam():
    '''
    Ncf 模型的配置参数
    '''
    us_shape=None;# 用户服务数量二元组
    hid_feat=0;# 隐含矩阵特征数
    hid_units=[]; # 网络各个隐层隐含单元数
    reg_p = 0.01;
    drop_p=0.001;


class NcfTraParm():
    '''
    Ncf 训练参数
    '''
    train_data=None;# ([[u,s]],[[y]]) 二元组
    test_data=None;# ([[u,s]],[[y]]) 二元组
    datasize=0;# 训练数据总长度
    batchsize=1;# 默认批长度
    epoch=0;# 迭代次数
    batch_size=1;# 训练数据总长度
    learn_rate=0;# 训练速度
    lr_decy_step=0;# 衰减步数
    lr_decy_rate=0; # 衰减率
    load_cache_rec=False;# 是否加载记录数据
    cache_rec_path='';# 缓存路径
    result_file_path=''
    summary_path = 'summary';

class NcfCreParam3D():
    '''
    Ncf 模型的配置参数
    '''
    ust_shape=None;# 用户服务数量二元组
    hid_feat=0;# 隐含矩阵特征数
    hid_units=[]; # 网络各个隐层隐含单元数
    reg_p = 0.01;
    drop_p=0.001;


class NcfTraParm3D():
    '''
    Ncf 训练参数
    '''
    train_data=None;# ([[u,s,t]],[[y]]) 二元组
    test_data=None;# ([[u,s,t]],[[y]]) 二元组
    datasize=0;# 训练数据总长度
    batchsize=1;# 默认批长度
    epoch=0;# 迭代次数
    batch_size=1;# 训练数据总长度
    learn_rate=0;# 训练速度
    lr_decy_step=0;# 衰减步数
    lr_decy_rate=0; # 衰减率
    load_cache_rec=False;# 是否加载记录数据
    cache_rec_path='';# 缓存路径
    result_file_path='';# 训练中测试结果输出
    summary_path=''; # 统计输出路径
    
    
    
    
