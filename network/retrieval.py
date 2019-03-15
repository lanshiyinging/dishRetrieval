

train_output_path = '../data/output/train_output'
test_output_path = '../data/output/test_output'

def main():
    train_output_file = open(train_output_path, 'r')
    test_output_file = open(test_output_path, 'r')
    train_hash_dataset = {}
    train_line = train_output_file.readline()
    while train_line:
        train_line = train_line.strip().strip('\n')
        train_line_list = train_line.split('\t')
        train_hash_dataset[train_line_list[0]] = train_line_list[1]
        train_line = train_output_file.readline()

    test_line = test_output_file.readline()
    test_result = {}
    while test_line:
        hm_dis_list = []
        test_line = test_line.strip().strip('\n')
        test_line_list = test_line.split('\t')
        test_pic_hashcode = test_line_list[1].split(',')
        for k, v in train_hash_dataset:
            candi_pic_hashcode = v.split(',')
            for code1, code2 in zip(candi_pic_hashcode, test_pic_hashcode):
                if int(code1) != int(code2):

        test_line = train_output_file.readline()