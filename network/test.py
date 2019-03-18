import numpy as np
import tensorflow as tf
import os

'''
train_dir = "../data/train_data/"
for label in os.listdir(train_dir):
    for pic in os.listdir(train_dir + label):
        #ret = os.system('identify -verbose ' + file_dir + file + ' | grep Interlace')
        filepath = train_dir + label + '/' + pic
        ret = os.popen('file ' + filepath)
        lines = ret.readlines()
        if 'progressive,' in str(lines):
            print(filepath)
            #progressive_to_baseline(file_dir, file)
'''

dir = "../data/all_data/"
total_num = 0
dic = {10: 0, 11: 0, 12: 0, 13: 0, 14: 0, 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0}
for label in os.listdir(dir):
    num = len(os.listdir(dir + label + '/'))
    total_num = total_num + num
    dic[num] += 1
print(dic)
print(total_num)
