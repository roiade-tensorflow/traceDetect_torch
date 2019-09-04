import torch.nn as  nn
import matplotlib.pyplot as plt
import  torch
import sys
sys.path.append('../../data/')
import  time
from data.myData import *

def MyPlotFuc(im,pre,label):
    import matplotlib.pyplot as plt
    import numpy as np
    ax=plt.subplot(131)
    data=np.where(im>0.8)
    ax.scatter(data[0],data[1],s=5)
    plt.xlim(0,128)
    plt.ylim(0,128)

    ax1=plt.subplot(132)
    data=np.where(pre>0.8)
    ax1.scatter(data[0],data[1],s=5)
    plt.xlim(0,128)
    plt.ylim(0,128)

    ax2=plt.subplot(133)
    data=np.where(label>0.8)
    ax2.scatter(data[0],data[1],s=5)

    plt.xlim(0,128)
    plt.ylim(0,128)
    plt.show()

def toCpu(data):
    data=data.cpu()
    data=data.detach().numpy()
    return data

class decodeNet(nn.Module):
    def __init__(self):
        super(decodeNet, self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv3d(1,64,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU())

        self.pool = nn.Sequential(
            nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,1))
        )
        self.conv2=nn.Sequential(
            nn.Conv3d(64,64,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,1,1)),
            nn.BatchNorm3d(64),
            nn.ReLU())
        self.upsimple = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.conv3=nn.Sequential(
            nn.Conv3d(64,64,kernel_size=(3,3,6),stride=(1,1,1),padding=(1,1,0)),
            nn.BatchNorm3d(64),
            nn.ReLU())
        self.conv4 = nn.Sequential(
            nn.Conv3d(64, 1, kernel_size=(3, 3, 7), stride=(1, 1, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(1),
            nn.ReLU())

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv2(out)
        out = self.conv2(out)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.pool(out)
        out = self.conv2(out)
        out = self.pool(out)
        out = self.upsimple(out)
        out = self.conv2(out)
        out = self.upsimple(out)
        out = self.conv2(out)
        out = self.upsimple(out)
        out = self.conv2(out)
        out = self.upsimple(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        # out = self.conv3(out)
        # out = self.conv3(out)
        return out


net=decodeNet().cuda()
net.load_state_dict(torch.load('net_save.pth'))


data_ = myDaDetection('../../', _txtName='128_5_20.txt', _width=128, _height=128, _time=5)
data_loader = data.DataLoader(data_, batch_size=1, shuffle=True)
batch_iterator = iter(data_loader)
for i in range(100):
    images, targets = next(batch_iterator)
    # MyPlotFuc(images[0,:,:,:])
    # MyPlotFuc(targets[0, :, :, :])
    images = images.float()
    targets = targets.float()
    images = images.unsqueeze(0)
    images = images.squeeze(-1)
    targets = targets.unsqueeze(0)
    images = images.cuda()
    targets = targets.cuda()
    images = Variable(images)
    targets = Variable(targets)

    out = net(images)
    out=toCpu(out)
    targets=toCpu(targets)
    images=toCpu(images)
    # print(out.size())
    dir=np.where(out<0.90)
    xx=np.ones_like(out)
    xx[dir[0],dir[1],dir[2],dir[3],dir[4]]=0

    yy=np.logical_and(out,images)
    MyPlotFuc(images[0,0,:,:,:],yy[0,0,:,:,:],targets[0,0,:,:,:])



    # print(out.size())
    # MyPlotFuc(out[0,0,:,:,:])
    # MyPlotFuc(targets[0,0,:,:,:])


