# -*- coding: utf-8 -*-
'''
Created on 2018年9月4日

@author: zwp12
'''



from tools import localload;



path = 'E:/work/Dataset/ws/localinfo/user_info.txt';


serve_loc = localload.load_userinfo(path);

for i in serve_loc:
    print(i,serve_loc[i]);




if __name__ == '__main__':
    pass