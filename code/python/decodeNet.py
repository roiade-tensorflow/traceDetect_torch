import torch.nn as nn
import torch
from torch.autograd import Variable

import sys
sys.path.append('../../data/')
from data.myData import *
import torch.utils.data as data
import torch.optim as optim
class MyLoss(nn.Module):
    def __init__(self):
        super().__init__()  # 没有需要保存的参数和状态信息

    def forward(self, pre, label):  # 定义前向的函数运算即可
        num = 1
        for i in range(len(torch.tensor(pre.size()))):
            num *= pre.size()[i]
        c = torch.pow(label - pre, 2)
        d = torch.pow((40) * (label - pre) * label, 2)
        f = torch.sum(c +d)
        g=torch.sum(torch.pow((label - pre) * label, 2))
        return f,g,torch.sum(c)





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
def MyPlotFuc(data):
    import matplotlib.pyplot as plt
    import numpy as np
    data=data.cpu()
    data=data.numpy()
    data=np.where(data>0)
    plt.xlim(0,128)
    plt.ylim(0,128)
    plt.scatter(data[0],data[1],s=5)
    plt.show()

# if __name__=='__main__':
net=decodeNet().cuda()
net.load_state_dict(torch.load('net_save.pth'))
opt_Adam = torch.optim.Adam(net.parameters(), lr=0.01, betas=(0.9, 0.99))
# loss_func=nn.CrossEntropyLoss()
# loss_func = nn.MSELoss()
loss_func=MyLoss()
#load data
data_ = myDaDetection('../../', _txtName='128_5_20.txt', _width=128, _height=128, _time=5)
data_loader = data.DataLoader(data_, batch_size=1, shuffle=True)
batch_iterator = iter(data_loader)
print(len(batch_iterator))

# for iteration in range(0, len(batch_iterator), 1):
net_save=net
lossmin=10000
lossAll=[]
for iteration in range(len(batch_iterator)):
    images, targets = next(batch_iterator)

    # MyPlotFuc(images[0,:,:,:])
    # MyPlotFuc(targets[0, :, :, :])
    images=images.float()
    targets=targets.float()

    images=images.unsqueeze(0)
    images=images.squeeze(-1)
    targets=targets.unsqueeze(0)

    images=images.cuda()
    targets=targets.cuda()

    images=Variable(images)
    targets=Variable(targets)

    # print(images.size(),targets.size())
    out=net(images)
    out=out.squeeze(0)
    targets=targets.squeeze(0)
    loss,dif,diff2=loss_func(out,targets)
    opt_Adam.zero_grad()
    loss.backward()
    opt_Adam.step()

    lossAll.append(loss)
    if iteration%10==0:
        print('loss:',loss,'diffrence:',dif,'diff2',diff2)
    if loss<lossmin:
        lossmin=loss
        net_save=net
    if iteration % 100 == 0:
        print('saving model ......')
        torch.save(net_save.state_dict(), 'net_save.pth')








