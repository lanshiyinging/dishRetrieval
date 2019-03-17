import tensorflow as tf
import numpy as np
import os
from PIL import Image
from PIL import ImageFile
import imghdr

# error--image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True
train_dir = "../data/train_data/"
test_dir = "../data/test_data/"
image_to_modify = []

def IsValidImage(filename):
    isValid = True
    try:
        Image.open(filename).verify()
    except:
        isValid = False
    return isValid

def is_valid_jpg(filename):
    with open(filename, 'rb') as f:
        f.seek(-2, 2)
        return f.read() == b'\xff\xd9'

def is_jpg(filename):
    try:
        i = Image.open(filename)
        return i.format == 'JPEG'
    except IOError:
        print("can't open file" + filename)
        return False

def check_jpg_pic(filename):
    for label in os.listdir(filename):
        for pic in os.listdir(filename + label):
            path = filename + label + '/' + pic
            image_c = tf.read_file(path)
            try:
                image = tf.image.decode_jpeg(image_c, channels=3)
            except:
                print(path)


'''
train_num = 0
for label in os.listdir(train_dir):
    for pic in os.listdir(train_dir+label):
        filename = train_dir + label + '/' + pic
        if not is_jpg(filename):
            print("not jpg file" + filename)
            image_to_modify.append(filename)
        if not is_valid_jpg(filename):
            print("break file" + filename)
            train_num += 1
        if not IsValidImage(filename):
            print("not valid pic" + filename)
'''

'''
for img in image_to_modify:
    filename_list = img.split('/')
    label = filename_list[-2]
    pic_name = filename_list[-1]
    prefix = "../data/train_fix_image/" + label + '/'
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    Image.open(img).convert('RGB').save(prefix+pic_name)
    if not is_jpg(prefix+pic_name):
        print("fail")
        break
'''

'''
test_num = 0
for pic in os.listdir(test_dir):
    filename = test_dir + pic
    if not is_jpg(filename):
        print("not jpg file" + filename)
        image_to_modify.append(filename)
    if not is_valid_jpg(filename):
        print("break file" + filename)
        test_num += 1
    if not IsValidImage(filename):
        print("not valid pic" + filename)

print(train_num)
print(test_num)
'''

'''
for img in image_to_modify:
    filename_list = img.split('/')
    pic_name = filename_list[-1]
    prefix = "../data/test_fix_image/"
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    # error --- raise IOError("cannot write mode %s as JPEG" % im.mode)
    Image.open(img).convert('RGB').save(prefix+pic_name)
    if not is_jpg(prefix+pic_name):
        print("fail")
        break
'''

if __name__ == '__main__':
    #check_jpg_pic(train_dir)
    for label in os.listdir(train_dir):
        for pic in os.listdir(train_dir + label):
            file_path = train_dir + label + '/' + pic
            print(is_valid_jpg(file_path))

