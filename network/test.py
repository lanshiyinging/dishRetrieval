import numpy as np
import tensorflow as tf
import os

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