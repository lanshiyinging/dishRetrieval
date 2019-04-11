import os

train_output_path = '../data/output_mmini_web/train_output.txt'
test_output_path = '../data/output_mmini_web/test_output.txt'
test_label_path = '../data/test_list_mmini.txt'
train_data_dir = '../data/train_data_mmini/'

train_output_path_runtime = '/root/lsy/dishRetrieval/data/output/train_output.txt'

def load_train_data(train_output_path):
    train_output_file = open(train_output_path, 'r')
    train_hash_dataset = {}
    train_label_dataset = {}
    train_line = train_output_file.readline()
    while train_line:
        train_line = train_line.strip().strip('\n')
        train_line_list = train_line.split('\t')
        train_label_dataset[train_line_list[0]] = train_line_list[1]
        train_hash_dataset[train_line_list[0]] = train_line_list[2]
        train_line = train_output_file.readline()
    return train_hash_dataset, train_label_dataset


def retrieval(quary_hashcode):
    train_hash_dataset, train_label_dataset = load_train_data(train_output_path_runtime)
    hm_dis_list = {}
    for k, v in train_hash_dataset.items():
        candi_pic_hashcode = v.split(',')
        temp_dis = 0
        for code1, code2 in zip(candi_pic_hashcode, quary_hashcode):
            if code1 != code2:
                temp_dis += 1
        hm_dis_list[k] = temp_dis
    sort_hm_dis = sorted(hm_dis_list.items(), key=lambda x: x[1])
    result = []
    for i in range(5):
        result.append(sort_hm_dis[i])
    return result


def get_test_label(filename):
    lable_dict = {}
    f = open(filename)
    line = f.readline()
    while line:
        line = line.strip().strip('\n')
        line_list = line.split('\t')
        lable_dict['../data/test_data_mmini/'+line_list[0]] = line_list[1]
        line = f.readline()
    return lable_dict


def evaluate(test_label, result, train_label_dataset):
    eval = {}
    total_num = 0
    for pic in os.listdir(train_data_dir + test_label):
        total_num += 1
    result_num = len(result)
    true_num = 0
    AP_5 = 0.0
    for i in range(result_num):
        candi_img_path = result[i][0]
        if int(train_label_dataset[candi_img_path]) == int(test_label):
            true_num += 1
            AP_5 = AP_5 + float(true_num)/(i+1)
    precision = float(true_num)/result_num
    recall = float(true_num)/total_num
    if true_num == 0:
        AP_5 = 0
    else:
        AP_5 = AP_5/true_num
    eval['precision'] = precision
    eval['recall'] = recall
    eval['AP_5'] = AP_5
    return eval

def main():
    train_output_file = open(train_output_path, 'r')
    test_output_file = open(test_output_path, 'r')
    train_hash_dataset = {}
    train_label_dataset = {}
    train_line = train_output_file.readline()
    while train_line:
        train_line = train_line.strip().strip('\n')
        train_line_list = train_line.split('\t')
        train_label_dataset[train_line_list[0]] = train_line_list[1]
        train_hash_dataset[train_line_list[0]] = train_line_list[2]
        train_line = train_output_file.readline()

    test_line = test_output_file.readline()
    test_label_dataset = get_test_label(test_label_path)
    MAP_5 = 0
    test_num = 0
    while test_line:
        hm_dis_list = {}
        result = []
        test_line = test_line.strip().strip('\n')
        test_line_list = test_line.split('\t')
        test_pic_hashcode = test_line_list[1].split(',')
        for k, v in train_hash_dataset.items():
            candi_pic_hashcode = v.split(',')
            temp_dis = 0
            for code1, code2 in zip(candi_pic_hashcode, test_pic_hashcode):
                if code1 != code2:
                    temp_dis += 1
            hm_dis_list[k] = temp_dis
        sort_hm_dis = sorted(hm_dis_list.items(), key=lambda x: x[1])
        for i in range(3):
            result.append(sort_hm_dis[i])
        eval = evaluate(test_label_dataset[test_line_list[0]], result, train_label_dataset)
        MAP_5 += eval['AP_5']
        with open("../data/test_result_mmini_web.txt", 'a') as f:
            f.write("%s\t%s\t[%s]\t%s\t%s\n" % (test_line_list[0], test_label_dataset[test_line_list[0]], test_line_list[1], str(result), str(eval)))
        test_num += 1
        test_line = test_output_file.readline()

    MAP_5 = MAP_5/test_num
    print('The MAP@5 is :' + str(MAP_5))


if __name__ == '__main__':
    main()

