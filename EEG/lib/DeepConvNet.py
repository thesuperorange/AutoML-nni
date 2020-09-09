import torch.nn as nn
from functools import reduce


class DeepConvNet(nn.Module):
    def __init__(self, activation="ELU"):
        # constructor
        super(DeepConvNet, self).__init__()

        self.activation = activation

        if activation is "ReLU":
            act_func = nn.ReLU()
        elif activation is "LeakyReLU":
            act_func = nn.LeakyReLU()
        else:
            act_func = nn.ELU()

        super(DeepConvNet, self).__init__()

        self.filter_list = [25, 50, 100, 200]
        self.conv0 = nn.Sequential(nn.Conv2d(1, self.filter_list[0], kernel_size=(1, 5)))

        self.conv1 = nn.Sequential(nn.Conv2d(self.filter_list[0], self.filter_list[0], kernel_size=(2, 1)),
                                   nn.BatchNorm2d(self.filter_list[0], eps=1e-05, momentum=0.1),
                                   act_func,
                                   nn.MaxPool2d(kernel_size=(1, 2)),
                                   nn.Dropout(p=0.5))

        for idx in range(len(self.filter_list)-1):  # conv2, conv3, conv4
            setattr(self,'conv'+str(idx+2),
                    nn.Sequential(nn.Conv2d(self.filter_list[idx], self.filter_list[idx+1], kernel_size=(1, 5)),
                    nn.BatchNorm2d(self.filter_list[idx+1], eps=1e-05, momentum=0.1),
                    act_func,
                    nn.MaxPool2d(kernel_size=(1, 2)),
                    nn.Dropout(p=0.5))
                    )

        # flatten
        in_feature_num = 200 * reduce(lambda x, _: round((x - 4) / 2), self.filter_list, 750)
        self.classify = nn.Sequential(nn.Linear(in_features=in_feature_num, out_features=2, bias=True))

    def forward(self, x):
        for i in range(len(self.filter_list)+1):
            x = getattr(self, 'conv' + str(i))(x)

        x = x.view(-1, self.classify[0].in_features)
        y = self.classify(x)
        return y
