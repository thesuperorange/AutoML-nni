import torch.nn as nn
from nni.nas.tensorflow.mutables import LayerChoice
from collections import OrderedDict

class EEGNet(nn.Module):
    def __init__(self, activation="ReLU"):
        # constructor
        super(EEGNet, self).__init__()

        self.activation = activation

        if activation is "ReLU":
            act_func = nn.ReLU()
        elif activation is "LeakyReLU":
            act_func = nn.LeakyReLU()
        else:
            act_func = nn.ELU()

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
