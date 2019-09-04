import torch
import torch.nn as nn
from torch.autograd import Variable
import  numpy as np
import torch.optim as optim
from torchstat import stat
import torchvision.models as models
base = [32,'C',128,'M',256,128*9]
class  TrackDetectionNet(nn.Module):
    def __init__(self,phase):
        super(TrackDetectionNet,self).__init__()
        self.phase=phase
        self.net=nn.ModuleList(baseNet(self.phase))

        # print(self.net)
    def forward(self,x):
        for i in range(len(self.net)-2):
            x=self.net[i](x)
        x=torch.squeeze(x,dim=4)

        x=self.net[i+1](x)
        x = self.net[i+1](x)
        i=i+1
        # print(self.net[i])
        x = self.net[i+1](x)
        return x


# base = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',512, 512, 512]
base = [32,'C',128,'M',256,128*9]
def baseNet(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))]
        elif v == 'C':
            layers += [nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1), ceil_mode=True)]
        else:
            conv3d = nn.Conv3d(in_channels, v, kernel_size=3, padding=(1,1,0))
            if batch_norm:
                layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv3d, nn.ReLU(inplace=True)]
            in_channels=v
    layers +=[nn.PixelShuffle(2,)]
    in_channels=72
    # conv2d = nn.Conv2d(in_channels, 9, kernel_size=1)
    layers += [nn.Conv2d(in_channels, 9, kernel_size=1)]

    return layers

if __name__=="__main__":
    from torch.autograd import Variable
    torch.cuda.empty_cache()
    xx=torch.zeros(size=(1,1,1024,1024,9))
    xx=Variable(xx)

    xx=xx.cuda()
    net=TrackDetectionNet(base)
    net=net.cuda()
    y=net(xx)

