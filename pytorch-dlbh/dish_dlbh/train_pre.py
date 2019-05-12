import os
import shutil
import argparse

import torch
import torch.nn as nn

from net import AlexNetPlusLatent

from torchvision import datasets, models, transforms
from torch.autograd import Variable
import torch.optim.lr_scheduler
import loadData


parser = argparse.ArgumentParser(description='Deep Hashing')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--epoch', type=int, default=100, metavar='epoch',
                    help='epoch')
parser.add_argument('--pretrained', type=int, default=49, metavar='pretrained_model',
                    help='loading pretrained model(default = None)')
parser.add_argument('--bits', type=int, default=24, metavar='bts',
                    help='binary bits')
parser.add_argument('--path', type=str, default='model_pre', metavar='P',
                    help='path directory')
args = parser.parse_args()

best_acc = 0
start_epoch = 1
trainloader, testloader = loadData.load_data()

net = AlexNetPlusLatent(args.bits)

use_cuda = torch.cuda.is_available()

if use_cuda:
    net.cuda()

softmaxloss = nn.CrossEntropyLoss().cuda()

optimizer4nn = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0.0005)

scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer4nn, milestones=[64], gamma=0.1)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        _, outputs = net(inputs)
        loss = softmaxloss(outputs, targets)
        optimizer4nn.zero_grad()

        loss.backward()

        optimizer4nn.step()

        train_loss += softmaxloss(outputs, targets).item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        print(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100*int(correct)/int(total), correct, total))
    return train_loss/(batch_idx+1)

def test():
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        _, outputs = net(inputs)
        loss = softmaxloss(outputs, targets)
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        print(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss/(batch_idx+1), 100*int(correct)/int(total), correct, total))
    acc = 100*int(correct) / int(total)
    print('Saving')
    if not os.path.isdir('{}'.format(args.path)):
        os.mkdir('{}'.format(args.path))
    torch.save(net.state_dict(), './{}/{}.model'.format(args.path, acc))

def main():
    if args.pretrained:
        net.load_state_dict(torch.load('./{}/{}'.format(args.path, args.pretrained)))
        for epoch in range(start_epoch, start_epoch+args.epoch):
            train(epoch)
            test()
            scheduler.step()
        #test()
    else:
        #if os.path.isdir('{}'.format(args.path)):
            #shutil.rmtree('{}'.format(args.path))
        for epoch in range(start_epoch, start_epoch+args.epoch):
            train(epoch)
            test()
            scheduler.step()


if __name__ == '__main__':
    main()
