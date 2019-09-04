import torch.nn  as  nn
import torch
class traceDecNet(nn.Module):
    def __init__(self,base,upsimple,nn2dphase):
        super(traceDecNet,self).__init__()
        self.baseNet=nn.ModuleList(netBase(base))
        self.upSimple=nn.ModuleList(upSimple(upsimple))
        self.NN2D=nn.ModuleList(MyNN2D(nn2dphase,i=320))


    def forward(self, input):
        for item in range(len(self.baseNet)):
            input=self.baseNet[item](input)
        # print('经过网络"base"后的网络大小：',input.size())
        input=torch.squeeze(input,4)
        for item in range ( len(self.upSimple)):
            input=self.upSimple[item](input)
        for item in range ( len(self.NN2D)):
            input=self.NN2D[item](input)
        input=input.permute(0,2,3,1)

        return input
def netBase(cfg, i=1, batch_norm=True):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool3d(kernel_size=(2,2,1), stride=(2,2,1))]
        elif v == 'C':
            layers += [nn.MaxPool3d(kernel_size=(2,2,1), stride=(2,2,1), ceil_mode=True)]
        else:
            conv3d = nn.Conv3d(in_channels, v, kernel_size=(5,5,3), padding=(2,2,0))
            if batch_norm:
                layers += [conv3d, nn.BatchNorm3d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv3d, nn.ReLU(inplace=True)]
            in_channels = v
    return layers
def upSimple(cfg):
    layers=[]
    for v in cfg:
        layers+=[nn.PixelShuffle(v)]
        # layers+=[nn.Upsample(v,)]
    return layers
def MyNN2D(cfg,i=320,batch_norm=True):
    layers = []
    in_channels = i
    for v in cfg:
        conv2d = nn.Conv2d(in_channels, v, kernel_size=(1,1), padding=(0,0))
        if batch_norm:
            layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
        else:
            layers += [conv2d, nn.ReLU(inplace=True)]
        in_channels = v
    return layers

#base网络用于将三维矩阵转换为一维
base = {
    '9': [6,  'M', 32, 'M', 64, 'M',64*9],
    '5': [256 ,'C',256*5],
}
upsimple={'9':[2,2,2],
          '5':[2]}
nn2dphase={'5':[160,64,32,5]}
if __name__=='__main__':
    import  torch
    from torch.autograd import Variable
    xx = torch.zeros(size=(1, 1, 128, 128, 5))
    xx=Variable(xx).cuda()
    net=traceDecNet(base['5'],upsimple['5'],nn2dphase['5']).cuda()
    print(net)
    print(xx.size())
    y=net(xx)
    print(y.size())