import os
import argparse

import numpy as np
from net import AlexNetPlusLatent

from timeit import time

import torch
import torch.nn as nn

from torchvision import datasets, models, transforms
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler
import loadData_mini


parser = argparse.ArgumentParser(description='Deep Hashing evaluate mAP')
parser.add_argument('--pretrained', type=str, default=32, metavar='pretrained_model',
                    help='loading pretrained model(default = None)')
parser.add_argument('--bits', type=int, default=12, metavar='bts',
                    help='binary bits')
parser.add_argument('--path', type=str, default='model_12', metavar='P',
                    help='path directory')
args = parser.parse_args()


def binary_output(dataloader):
    net = AlexNetPlusLatent(args.bits)
    net.load_state_dict(torch.load('./{}/{}.model'.format(args.path, args.pretrained)))
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        net.cuda()
    #full_batch_output = torch.cuda.FloatTensor()
    #full_batch_label = torch.cuda.LongTensor()
    full_batch_output = torch.FloatTensor()
    full_batch_label = torch.LongTensor()
    net.eval()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs, _ = net(inputs)
        full_batch_output = torch.cat((full_batch_output, outputs.data), 0)
        full_batch_label = torch.cat((full_batch_label, targets.data), 0)
    return torch.round(full_batch_output), full_batch_label

def precision(train_binary, train_label, test_binary, test_label):
    train_binary = train_binary.cpu().numpy()
    train_binary = np.asarray(train_binary, np.int32)
    train_label = train_label.cpu().numpy()
    test_binary = test_binary.cpu().numpy()
    test_binary = np.asarray(test_binary, np.int32)
    test_label = test_label.cpu().numpy()
    classes = np.max(test_label) + 1
    for i in range(classes):
        if i == 0:
            test_sample_binary = test_binary[np.random.RandomState(seed=i).permutation(np.where(test_label==i)[0])[:2]]
            test_sample_label = np.array([i]).repeat(2)
            continue
        else:
            test_sample_binary = np.concatenate([test_sample_binary, test_binary[np.random.RandomState(seed=i).permutation(np.where(test_label==i)[0])[:2]]])
            test_sample_label = np.concatenate([test_sample_label, np.array([i]).repeat(2)])
    query_times = test_sample_binary.shape[0]
    trainset_len = train_binary.shape[0]
    AP = np.zeros(query_times)
    precision_radius = np.zeros(query_times)
    Ns = np.arange(1, trainset_len + 1)
    sum_tp = np.zeros(trainset_len)
    for i in range(query_times):
        print('Query ', i+1)
        query_label = test_sample_label[i]
        query_binary = test_sample_binary[i,:]
        query_result = np.count_nonzero(query_binary != train_binary, axis=1)    #don't need to divide binary length
        sort_indices = np.argsort(query_result)
        buffer_yes = np.equal(query_label, train_label[sort_indices]).astype(int)
        P = np.cumsum(buffer_yes) / Ns
        precision_radius[i] = P[np.where(np.sort(query_result)>2)[0][0]-1]
        AP[i] = np.sum(P * buffer_yes) /sum(buffer_yes)
        sum_tp = sum_tp + np.cumsum(buffer_yes)
    precision_at_k = sum_tp / Ns / query_times
    #index = [100, 200, 400, 600, 800, 1000]
    index = [5, 10]
    index = [i - 1 for i in index]
    print('precision at k:', precision_at_k[index])
    np.save('precision_at_k', precision_at_k)
    print('precision within Hamming radius 2:', np.mean(precision_radius))
    map = np.mean(AP)
    print('mAP:', map)



def main():
    result_path = "result_mini_12"
    train_binary_path = './%s/train_binary' % result_path
    train_label_path = './%s/train_label' % result_path
    test_binary_path = './%s/test_binary' % result_path
    test_label_path = './%s/test_label' % result_path
    if os.path.exists(train_binary_path) and os.path.exists(train_label_path) and \
       os.path.exists(test_binary_path) and os.path.exists(test_label_path) and args.pretrained == 0:
        train_binary = torch.load(train_binary_path)
        train_label = torch.load(train_label_path)
        test_binary = torch.load(test_binary_path)
        test_label = torch.load(test_label_path)
    
    else:
        trainloader, testloader = loadData_mini.load_data()
        train_binary, train_label = binary_output(trainloader)
        test_binary, test_label = binary_output(testloader)
        if not os.path.isdir(result_path):
            os.mkdir(result_path)
        torch.save(train_binary, train_binary_path)
        torch.save(train_label, train_label_path)
        torch.save(test_binary, test_binary_path)
        torch.save(test_label, test_label_path)
    
    
    precision(train_binary, train_label, test_binary, test_label)
    

if __name__ == '__main__':
    main()
