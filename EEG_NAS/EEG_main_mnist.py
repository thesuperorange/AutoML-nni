"""
A deep MNIST classifier using convolutional layers.

This file is a modification of the official pytorch mnist example:
https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import os
import argparse
import logging
from collections import OrderedDict
from dataloader import read_bci_data
import torch.utils.data as data

import nni
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from nni.nas.pytorch.mutables import LayerChoice, InputChoice
from nni.nas.pytorch.classic_nas import get_and_apply_next_architecture


logger = logging.getLogger('EEGNet ClassicNAS')

class EEGNet(nn.Module):
    def __init__(self):
        # constructor
        super(EEGNet, self).__init__()


        act_func = nn.ReLU()

        # first convolution
        self.conv1 =  LayerChoice(OrderedDict([
            ("conv51", nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False)),
            ("conv25", nn.Conv2d(1, 16, kernel_size=(1, 25), stride=(1, 1), padding=(0, 25), bias=False))
        ]), key='first_conv')

        self.firstconv = nn.Sequential(


            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))

        # depth-wise convolution
        self.depthwiseconv = nn.Sequential(nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
                                           nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                           act_func,
                                           nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
                                           nn.Dropout(p=0.25))

        # separable convolution
        self.separableconv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            act_func,
            nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0),
            nn.Dropout(p=0.25))

        # classify  (FC)
        self.classify = nn.Sequential(nn.Linear(in_features=736, out_features=2, bias=True))

    def forward(self, x):
        x = self.conv1(x)
        x = self.firstconv(x)
        x = self.depthwiseconv(x)
        x = self.separableconv(x)
        # FC Layer
        # print("feature size={}".format(self.classify[0].in_features))
        x = x.view(-1, self.classify[0].in_features)  # flatten to 736
        y = self.classify(x)
        return y



def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).long()
        optimizer.zero_grad()
        output = model(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
#        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args['log_interval'] == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
 #           test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

  #  test_loss /= len(test_loader.dataset)

    accuracy = 100. * correct / len(test_loader.dataset)

   # logger.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    #    test_loss, correct, len(test_loader.dataset), accuracy))

    return accuracy

def load_data(train_data, train_label,BATCH_SIZE):
    x_train_tensor = torch.from_numpy(train_data).float()
    y_train_tensor = torch.from_numpy(train_label)

    train_dataset = data.TensorDataset(x_train_tensor, y_train_tensor)

    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        # num_workers=2,
    )

    return train_loader


def main(args):
    use_cuda = not args['no_cuda'] and torch.cuda.is_available()

    torch.manual_seed(args['seed'])

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    data_dir = args['data_dir']

#    train_loader = torch.utils.data.DataLoader(
#        datasets.MNIST(data_dir, train=True, download=True,
#                       transform=transforms.Compose([
#                           transforms.ToTensor(),
#                           transforms.Normalize((0.1307,), (0.3081,))
#                       ])),
#        batch_size=args['batch_size'], shuffle=True, **kwargs)
#    test_loader = torch.utils.data.DataLoader(
#        datasets.MNIST(data_dir, train=False, transform=transforms.Compose([
#            transforms.ToTensor(),
#            transforms.Normalize((0.1307,), (0.3081,))
#        ])),
#        batch_size=1000, shuffle=True, **kwargs)
    BATCH_SIZE=args['batch_size'] 
    train_data, train_label, test_data, test_label = read_bci_data()
    train_loader = load_data(train_data, train_label,BATCH_SIZE)
    test_loader = load_data(test_data, test_label,BATCH_SIZE)


    hidden_size = args['hidden_size']

    model = EEGNet().to(device)
    get_and_apply_next_architecture(model)
#    optimizer = optim.SGD(model.parameters(), lr=args['lr'],
#                          momentum=args['momentum'])
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])  # , weight_decay=1e-2

    for epoch in range(1, args['epochs'] + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_acc = test(args, model, device, test_loader)

        if epoch < args['epochs']:
            # report intermediate result
            nni.report_intermediate_result(test_acc)
            logger.debug('test accuracy %g', test_acc)
            logger.debug('Pipe send intermediate result done.')
        else:
            # report final result
            nni.report_final_result(test_acc)
            logger.debug('Final result is %g', test_acc)
            logger.debug('Send final result done.')


def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument("--data_dir", type=str,
                        default='./data', help="data directory")
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument("--hidden_size", type=int, default=512, metavar='N',
                        help='hidden layer size (default: 512)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log_interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging training status')

    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        params = vars(get_params())
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
