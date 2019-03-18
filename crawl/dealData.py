import os
import random
import shutil
from PIL import Image
from PIL import ImageFile

# error--image file is truncated
ImageFile.LOAD_TRUNCATED_IMAGES = True

map_file = 'dishname2.txt'
data_dir = '../data/all_data/'
train_data_dir = '../data/train_data/'
#val_data_dir = '../data/val_data/'
test_data_dir = '../data/test_data/'
os.makedirs(test_data_dir)
rank1 = 0
#rank2 = 0
rank3 = 0

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
        return f.read() == '\xff\xd9'

def is_jpg(filename):
    try:
        i = Image.open(filename)
        return i.format == 'JPEG'
    except IOError:
        print("can't open file" + filename)
        return False


f = open(map_file, 'r')
line = f.readline()
get_2_list = [12, 13]
get_3_list = [14, 15, 16, 17, 20]
get_4_list = [18, 19]
while line:
    line = line.strip().strip('\n')
    dish_no = line.split('\t')[0]
    dish_name = line.split('\t')[1]
    old_path = data_dir + dish_name + '/'
    new_path = data_dir + dish_no + '/'
    if not os.path.exists(old_path):
        print(dish_name)
        line = f.readline()
        continue
    #os.rename(old_path, new_path)
    os.makedirs(train_data_dir + dish_no + '/')
    #os.makedirs(val_data_dir + dish_no + '/')
    file_list = os.listdir(old_path)
    total_num = len(file_list)
    if total_num in get_2_list:
        test_num = 2
    elif total_num in get_3_list:
        test_num = 3
    else:
        test_num = 4
    #for i, j in zip(range(val_num), range(test_num)):
    for i in range(test_num):
	
        #vfile = random.choice(file_list)
        #file_list.remove(vfile)
        #to_path = "%s%s/Validate_%04d.jpg" % (val_data_dir, dish_no, rank2)
        #shutil.copyfile(old_path+vfile, to_path)
       
        tfile = random.choice(file_list)
        file_list.remove(tfile)
        to_path = "%sTest_%04d.jpg" % (test_data_dir, rank3)
        shutil.copyfile(old_path + tfile, to_path)
        '''
        if is_jpg(old_path + tfile):
            shutil.copyfile(old_path + tfile, to_path)
        else:
            continue
            #Image.open(old_path + tfile).convert('RGB').save(to_path)
        '''

        with open('../data/test_list.txt', 'a') as ft:
            ft.write("Test_%04d.jpg\t%s\n" %(rank3, dish_no))
        #rank2 += 1

        rank3 += 1

    for file in file_list:
        to_path = "%s%s/Train_%04d.jpg" % (train_data_dir, dish_no, rank1)
        shutil.copyfile(old_path + file, to_path)
        '''
        if is_jpg(old_path + file):
            shutil.copyfile(old_path + file, to_path)
        else:
            continue
            #Image.open(old_path + file).convert('RGB').save(to_path)
        '''

        with open('../data/train_list.txt', 'a') as tf:
            tf.write("Train_%04d\t%s\n" % (rank1, dish_no))
        rank1 += 1
    line = f.readline()
