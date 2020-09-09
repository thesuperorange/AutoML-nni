from DataLoader import RespiratoryLoader

import torchvision
import argparse

import logging
import nni
import os
import time
import calendar
import torch
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import os
from sys import stdout
import json
logger = logging.getLogger('respiratory_pytorch_AutoML')

def run(device, train_dataloader, test_dataloader, net, optimizer, criterion,EPOCH_NUM):
    train_acc_trend = []
    test_acc_trend = []
    fo = open(output_folder_name+'/{}_{}_accuracy.log'.format(NET_NAME, PRETRAINED), "w")
    for epoch in range(1, EPOCH_NUM + 1):
        running_loss = 0.0
        print("Epoch {}:".format(epoch))
        # fo.write("Epoch {}:\n".format(epoch))
        # training mode
        net.train()
        train_corr_count = 0
        for idx, (inputs, labels) in enumerate(train_dataloader):
            if idx % 20 == 0:
                stdout.write("\rBatch %i of %i" % (idx, len(train_dataloader)))
                stdout.flush()

            inputs = inputs.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()
            outputs = net(inputs)
            print(outputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # for output loss and accuracy
            running_loss += loss
            pred = torch.max(outputs, 1)[1]  # max index or row
            train_corr_count += pred.eq(labels.data).sum().item()  # .sum().data.cpu().item()

        acc_percentage_train = 100. * train_corr_count / len(train_dataloader.dataset)
        
        
        logger.info("train accuracy {}, loss= {}".format(acc_percentage_train, running_loss))
        # fo.write("train accuracy {}\n".format(acc_percentage))
        train_acc_trend.append(acc_percentage_train)

        # !!! put the network in evaluation mode
        net.eval()

        # save model
        save_name = os.path.join(output_folder_name, '{}_{}_{}.pth'.format(NET_NAME, epoch, PRETRAINED))
        torch.save({
            'epoch': epoch,
            'model': net.module.state_dict() if mGPUS else net.state_dict(),
            'optimizer': optimizer.state_dict()
        }, save_name)
        print('save model: {}'.format(save_name))

        # -----------------------------------------

        test_corr_count = 0
        with torch.no_grad():
            # training data
            # for idx, (inputs, labels) in enumerate(train_dataloader):
            #     inputs = inputs.to(device)
            #     labels = labels.to(device).long()
            #     outputs = net(inputs)
            #     pred = torch.max(outputs, 1)[1]  # max index or row
            #     train_corr_count += pred.eq(labels.data).sum().item()  # .sum().data.cpu().item()
            #
            # acc_percentage = 100. * train_corr_count / len(train_dataloader.dataset)
            #
            # print("train accuracy {}".format(acc_percentage))
            # fo.write("train accuracy {}\n".format(acc_percentage))
            # train_acc_trend.append(acc_percentage)

            # testing data
            for idx, (inputs, labels) in enumerate(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device).long()
                outputs = net(inputs)
                pred = torch.max(outputs, 1)[1]  # max index or row
                test_corr_count += pred.eq(labels.data).sum().item()

            acc_percentage_test = 100. * test_corr_count / len(test_dataloader.dataset)
            logger.info("test accuracy {}".format(acc_percentage_test))
     
            test_acc_trend.append(acc_percentage_test)
            nni.report_intermediate_result(acc_percentage_test)
            logger.debug('Pipe send intermediate result done.')

        fo.write("{},{},{},{}\n".format(epoch, acc_percentage_train, acc_percentage_test,running_loss))
    acc_trend = {'train': train_acc_trend, 'test': test_acc_trend}
    return acc_trend
def get_params():
    ap = argparse.ArgumentParser()
    ap.add_argument("-b", "--batch", required=False, default=4, help="batch size")
    ap.add_argument("-l", "--lr", required=False, default=1e-3, help="learning rate")
    ap.add_argument("-e", "--epoch", required=False,default=10,  help="epoch number")
    ap.add_argument("-d", "--decay", required=False, default=5e-4, help="optimizer decay")
    ap.add_argument("-p", "--pretrained", type=str2bool, default=True, help="pretrained=True/False") 
    ap.add_argument('-g', dest='mGPUs', help='whether use multiple GPUs', action='store_true')

    args, _ = ap.parse_known_args()
    return args

def main(img_root, csv_path,args):
    train_dataset = RespiratoryLoader(img_root, csv_path, 'train')
    test_dataset = RespiratoryLoader(img_root, csv_path, 'val')

    train_loader = data.DataLoader(train_dataset, batch_size=int(args['batch']), shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=int(args['batch']), shuffle=False)

    if torch.cuda.is_available():
        print("use cuda")
        device = torch.device('cuda')  # current cuda device
    #        model.cuda()

    else:
        device = torch.device('cpu')

    model = torchvision.models.resnet18(pretrained=PRETRAINED)
    print(model)
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    # model.avgpool = nn.AdaptiveAvgPool2d(1,1)  # not working in torchvision 0.2
    in_feature = model.fc.in_features
    print(in_feature)
    model.fc = nn.Linear(in_feature, NUM_CLASSES)

    #     model = torchvision.models.vgg16(pretrained=True)
    #     print(model.classifier[6].out_features)
    #     model.classifier[6].out_features=NUM_CLASSES

    # model.float()
    if mGPUS:
        model = nn.DataParallel(model)
    model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=float(args['lr']), weight_decay=float(args['decay']))  # , weight_decay=1e-2
    # optimizer = nn.DataParallel(optimizer)  # back propagation also use multiple GPU

    criterion = nn.CrossEntropyLoss()

    # !!! put the network in evaluation mode
    # model.eval()

    # acc_trend = run_simple(device, train_loader, test_loader, model, optimizer, criterion, output_folder, net_name)

    acc_trend = run(device, train_loader, test_loader, model, optimizer, criterion,int(args['epoch']))
    nni.report_final_result(acc_trend["test"][-1])
    logger.debug('Final result is %g', acc_trend["test"][-1])

    print("train acc: {}, test acc:{}".format(acc_trend["train"][-1], acc_trend["test"][-1]))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(get_params())
        params.update(tuner_params)
        print(params)
        mGPUS = params['mGPUs']

        output_folder_name = 'output/'+str (calendar.timegm(time.gmtime()))
        if not os.path.exists(output_folder_name):
                os.makedirs(output_folder_name)
        with open(output_folder_name+'/params.json', 'w') as outfile:
            json.dump(params, outfile)
        PRETRAINED = params['pretrained']  
        NUM_CLASSES = 8
        IMG_PATH = '../../dataset/output_img_new'
        #OUTPUT_FOLDER = "model"
        NET_NAME = "res18"
        csv_path = IMG_PATH#'.'
        main(IMG_PATH,csv_path,params)
    except Exception as exception:
        logger.exception(exception)
        raise


    

    


