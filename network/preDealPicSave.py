#import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

data_dir = "../data/all_data/"
deal_data_dir = "../data/all_data_deal/"
#img_path = '630.jpg'
#image_raw_data = tf.gfile.FastGFile(img_path, 'rb').read()
with tf.Session() as sess:
    for label in os.listdir(data_dir):
        for pic in os.listdir(data_dir + label):
            if os.path.exists(deal_data_dir+label+'/'+pic):
                continue
            img_path = data_dir + label + '/' + pic
            image_raw_data = tf.gfile.FastGFile(img_path, 'rb').read()
            img_data = tf.image.decode_jpeg(image_raw_data)
            image = tf.image.central_crop(img_data, 0.5)
            image = tf.image.resize_images(image, [32, 32], method=0)
            image = np.asarray(image.eval(), dtype='uint8')
            encoded_image = tf.image.encode_jpeg(image)
            if not os.path.exists(deal_data_dir + label):
                os.makedirs(deal_data_dir + label)
            with tf.gfile.GFile(deal_data_dir+label+'/'+pic, 'wb') as f:
                f.write(encoded_image.eval())

