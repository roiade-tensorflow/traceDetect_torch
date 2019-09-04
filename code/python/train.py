import sys
sys.path.append('../../data/')

import torch
from python.eCode import *
import importlib
from data.myData import *
from torch.autograd import Variable
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from  python.ImLoss import *
#加载网络
net = traceDecNet(base['5'], upsimple['5'],nn2dphase['5'])
if torch.cuda.is_available():
    net=net.cuda()
#生成数据集
data_=myDaDetection('../../',_txtName='128_5.txt',_width=128,_height=128,_time=5)
data_loader = data.DataLoader(data_, batch_size=8,shuffle=True)
batch_iterator = iter(data_loader)
# 神经网络优化器，主要是为了优化我们的神经网络
optimizer = optim.SGD(net.parameters(), lr=0.1)
opt_Adam= torch.optim.Adam(net.parameters(), lr=LR, betas=(0.9, 0.99))
# criterion =nn.NLLLoss()
criterion =My_loss()
for iteration in range(0,len(batch_iterator),1):
    images, targets = next(batch_iterator)
    images=images.unsqueeze(1)
    images = images.squeeze(-1).float()
    targets=targets.float()

    if torch.cuda.is_available():
    # if False:
        images=Variable(images.cuda())
        targets=Variable(targets.cuda())
    else:
        images=Variable(images)
        targets=Variable(targets)
    out = net(images)
    # print(out.size(),targets.size())
    optimizer.zero_grad()
    loss=criterion(out,targets)
    loss.backward()
    optimizer.step()
    if iteration%10==0:
        print(loss)

print('===> Saving models...')
# torch.save(net.state_dict(), 'net.pth')
torch.save(net,'net.pth')

    # print(images.size(),targets.size())
# model = TheModelClass(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()