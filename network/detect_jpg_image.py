import numpy as np
import os
from PIL import Image
from PIL import ImageFile

# error--image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True
train_dir = "../data/train_data/"
test_dir = "../data/test_data/"
image_to_modify = []

def is_jpg(filename):
    try:
        i = Image.open(filename)
        return i.format == 'JPEG'
    except IOError:
        print("can't open file" + filename)
        return False
'''
for label in os.listdir(train_dir):
    for pic in os.listdir(train_dir+label):
        filename = train_dir + label + '/' + pic
        if not is_jpg(filename):
            print("not jpg file" + filename)
            image_to_modify.append(filename)

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

for pic in os.listdir(test_dir):
    filename = test_dir + pic
    if not is_jpg(filename):
        print("not jpg file" + filename)
        image_to_modify.append(filename)

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