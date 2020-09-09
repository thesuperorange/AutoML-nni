import _init_paths

import matplotlib.pyplot as plt
from dataloader import read_bci_data
from EGGNet import EEGNet
from DeepConvNet import DeepConvNet
import torch.utils.data as data
import torch.optim as optim
import torch.nn as nn
import torch
import argparse
import nni
import logging
logger = logging.getLogger('EGG')


def run(device, train_dataloader, test_dataloader, net, optimizer, criterion):
    train_acc_trend = []
    test_acc_trend = []
    for epoch in range(EPOCH_NUM):

        # print("Epoch {}:".format(epoch))
        # training
        net.train()
        for idx, (inputs, labels) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device).long()

            optimizer.zero_grad()
            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        net.eval()

        train_corr_count = 0
        test_corr_count = 0
        with torch.no_grad():
            # training data
            for idx, (inputs, labels) in enumerate(train_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device).long()

                with torch.no_grad():
                    outputs = net(inputs)

                pred = torch.max(outputs, 1)[1]  # max index or row
                train_corr_count += pred.eq(labels.data).sum().item()  # .sum().data.cpu().item()

            acc_percentage = 100. * train_corr_count / len(train_dataloader.dataset)
            # print("accuracy {}".format(acc_percentage))
            train_acc_trend.append(acc_percentage)

            # testing data
            for idx, (inputs, labels) in enumerate(test_dataloader):

                inputs = inputs.to(device)
                labels = labels.to(device).long()
                outputs = net(inputs)
                pred = torch.max(outputs, 1)[1]  # max index or row
                test_corr_count += pred.eq(labels.data).sum().item()

            acc_percentage = 100. * test_corr_count / len(test_dataloader.dataset)
            print("test accuracy {}".format(acc_percentage))
            test_acc_trend.append(acc_percentage)
            nni.report_intermediate_result(acc_percentage)

    acc_trend = {'train': train_acc_trend, 'test': test_acc_trend}
    nni.report_final_result(acc_trend["test"][-1])
    logger.debug('Final result is %g', acc_trend["test"][-1])

    return acc_trend


def load_data(train_data, train_label):
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


def main():
    train_data, train_label, test_data, test_label = read_bci_data()
    train_dataloader = load_data(train_data, train_label)
    test_dataloader = load_data(test_data, test_label)

    #activation_list = ["LeakyReLU", "ELU", "ReLU"]
    #model_list = ["EEG", "DeepConv"]

    criterion = nn.CrossEntropyLoss()

#    fo = open("output_accuracy.txt", "a")
    # fo.write("model,batch,lr,epoch,activation,train_acc,test_acc\n")

#    for model_name in model_list:

#        plt.figure()
#        for activation in activation_list:
    if MODEL_NAME is "DeepConv":
        net = DeepConvNet(ACTIVATION)
    else:
        net = EEGNet(ACTIVATION)

    print(net)
    if torch.cuda.is_available():
        print("use cuda")
        device = torch.device('cuda')  # current cuda device
        net.cuda()

    else:
        device = torch.device('cpu')

            # net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=LR)  # , weight_decay=1e-2
    acc_trend = run(device, train_dataloader, test_dataloader, net, optimizer, criterion)
#    fo.write("{},{},{},{},{},{},{}\n".format(model_name, BATCH_SIZE, LR, EPOCH_NUM, ACTIVATION,
#                                                     acc_trend["train"][-1], acc_trend["test"][-1]))

            # print("model: {}, activation:{}, accuracy-- train: {}, test: {}".format(model_name, activation, ))
 #           plt.plot(acc_trend['train'], label=activation + '_train')
 #           plt.plot(acc_trend['test'], label=activation + '_test')

#        plt.legend(loc='lower right')
#        plt.title("Activation function comparision ("+model_name+")")
#        plt.ylabel("Accuracy(%)")
#        plt.xlabel("Epoch")
#        image_output_name = "{}_{}_{:e}_{}_acc.png".format(model_name, BATCH_SIZE, LR, EPOCH_NUM)
#        plt.savefig(image_output_name)

#    fo.close()


def get_params():
    ap = argparse.ArgumentParser()
    ap.add_argument("-b", "--batch", required=False, default=64, help="batch size")
    ap.add_argument("-l", "--lr", required=False, default=1e-2, help="learning rate")
    ap.add_argument("-e", "--epoch", required=False, default=150, help="epoch number")
    ap.add_argument("-a", "--activation", required=False, default="ELU", help="activation function")
    ap.add_argument("-m", "--modelname", required=False, default="EEG", help="EEG or DeepConv")


    args, _ = ap.parse_known_args()
    return args


if __name__ == '__main__':

    # draw training data
    # print(train_data.shape)
    #
    # plt.figure()
    # plt.subplot( 2, 1, 1)
    # plt.title('S4b train data', fontsize=18)
    # plt.plot(train_data[0][0][0])
    # plt.subplot(2, 1, 2)
    # plt.plot(train_data[0][0][1])

    # plt.show()
    try: 
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(get_params())
        params.update(tuner_params)
        print(params)
#    args = vars(ap.parse_args())
        BATCH_SIZE = int(params['batch'])
        LR = float(params['lr'])
        EPOCH_NUM = int(params['epoch'])
        ACTIVATION = params['activation']
        MODEL_NAME = params['modelname']
        main()
    except Exception as exception:
        logger.exception(exception)
        raise
