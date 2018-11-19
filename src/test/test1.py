# -*- coding: utf-8 -*-
'''
Created on 2018年8月4日

@author: zwp12
'''


import numpy as np;

from math import radians, cos, sin, asin, sqrt
 
def haversine( lat1, lon1, lat2,lon2,r=6371): # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    """
    # 将十进制度数转化为弧度
    lon1,lat1,lon2, lat2 = map(np.radians, [lon1,lat1,lon2, lat2])
    # haversine公式
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = r # 地球平均半径，单位为公里
    return c * r # 输出为公里


 
def haversine2(lat1,lon1, lat2,lon2): # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
 
    # haversine公式
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 20 # 地球平均半径，单位为公里
    return c * r 


a = np.random.uniform(-90,90,size=(5,2));
b = np.random.uniform(-90,90,size=(2,));
print(a,b);
dis = haversine(a[:,0],a[:,1],b[0],b[1]);
print(dis);


print(haversine2(a[1,0],a[1,1],b[0],b[1]))






if __name__ == '__main__':
    pass