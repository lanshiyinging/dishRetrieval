import os
import random
import shutil


map_file = 'dishname2.txt'
data_dir = 'data/'
train_data_dir = 'train_data/'
val_data_dir = 'val_data/'
test_data_dir = 'test_data/'
os.mkdir(test_data_dir)
rank1 = 0
rank2 = 0
rank3 = 0
f = open(map_file, 'r')
line = f.readline()
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
    os.makedirs(val_data_dir + dish_no + '/')
    file_list = os.listdir(old_path)
    total_num = len(file_list)
    val_num = int(total_num * 0.2)
    test_num = val_num
    for i, j in zip(range(val_num), range(test_num)):
        vfile = random.choice(file_list)
        file_list.remove(vfile)
        to_path = "%s%s/Validate_%04d.jpg" % (val_data_dir, dish_no, rank2)
        shutil.copyfile(old_path+vfile, to_path)

        tfile = random.choice(file_list)
        file_list.remove(tfile)
        to_path = "%sTest_%04d.jpg" % (test_data_dir, rank3)
        shutil.copyfile(old_path + tfile, to_path)

        with open('val_list.txt', 'a') as vf:
            vf.write("Validate_%04d.jpg\t%s\n" %(rank2, dish_no))
        rank2 += 1
        rank3 += 1

    for file in file_list:
        to_path = "%s%s/Train_%04d.jpg" % (train_data_dir, dish_no, rank1)
        shutil.copyfile(old_path + file, to_path)
        with open('train_list.txt', 'a') as tf:
            tf.write("Train_%04d\t%s\n" % (rank1, dish_no))
        rank1 += 1
    line = f.readline()
