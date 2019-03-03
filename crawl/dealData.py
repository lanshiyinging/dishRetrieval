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
        shutil.copyfile(old_path+vfile, val_data_dir + dish_no + '/' + str(rank2) + '.jpg')
        tfile = random.choice(file_list)
        file_list.remove(tfile)
        shutil.copyfile(old_path + tfile, test_data_dir + str(rank3) + '.jpg')
        with open('val_list.txt', 'a') as vf:
            vf.write("%d\t%s\n" %(rank2, dish_no))
        rank2 += 1
        rank3 += 1
    for file in file_list:
        shutil.copyfile(old_path + file, train_data_dir + dish_no + '/' + str(rank1) + '.jpg')
        with open('train_list.txt', 'a') as tf:
            tf.write("%d\t%s\n" %(rank1, dish_no))
        rank1 += 1
    line = f.readline()
