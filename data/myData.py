import  os.path as osp
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from torch.autograd import Variable


import numpy as np
class myDaDetection(data.Dataset):
    def __init__(self,root,_txtName='dataName.txt',_width=1024,_height=1024,_time=9):
        self._txtName=_txtName
        self._width=_width
        self._height=_height
        self._time=_time

        self.root=root
        self.ids=list()
        self._impath = osp.join('%s', 'Images', '%s')
        self._anapath=osp.join('%s', 'Annatations','trace', '%s')
        rootPath=osp.join(self.root,'data')
        for line in open(osp.join(rootPath,'Main',self._txtName),'r'):
            self.ids.append((rootPath, line.strip()))  # 将图片路径存入ids
    def __getitem__(self, index):
        img_id = self.ids[index]
        im=torch.load(self._impath % img_id)
        im=torch.tensor(im)
        # im=torch.clamp(im,0,1)
        ana=torch.load(self._anapath % img_id)

        image=torch.zeros(size=(self._width,self._height,self._time,1),dtype=im.dtype)
        im[:,0:2]=torch.clamp(im[:,0:2],0,1)
        im[:,0:1]=self._width*im[:,0:1]
        im[:,1:2]=self._height*im[:,1:2]
        im=im.long()
        image[im[:,0],im[:,1],im[:,2],0]=1

        annatition = torch.zeros(size=(self._width, self._height,self._time), dtype=im.dtype)
        ana[:, 0:2] = torch.clamp(ana[:, 0:2], 0, 1)
        ana[:, 0:1] = self._width * ana[:, 0:1]
        ana[:, 1:2] = self._height * ana[:, 1:2]
        ana = ana.long()
        annatition[ana[:, 0],ana[:, 1], ana[:, 2] ] = 1
        return image , annatition


    def __len__(self):
        return (len(self.ids))
    def plotFuc(self,image,model='train'):
        if model=='train':
            dirValue=np.where(image>0)
            print(dirValue)
            plt.scatter(dirValue[0],dirValue[1],s=1)
            plt.xlim(0,self._width)
            plt.ylim(0,self._height)
            plt.show()
        elif model=='label':
            dirValue=np.where(image>0)
            print(dirValue)
            plt.scatter(dirValue[1],dirValue[2],s=1)
            plt.xlim(0,self._width)
            plt.ylim(0,self._height)
            plt.show()


if __name__ =="__main__":
    data_=myDaDetection('../',_txtName='128_5.txt',_width=128,_height=128,_time=5)
    data_loader = data.DataLoader(data_, 2,)
    a=iter(data_loader)
    im,la=next(a)
    print(la.size())
    print(im.size())
    for i in range(im.size()[0]):
        data_.plotFuc(im[i])
        data_.plotFuc(la[i],'train')
        # data_.plotFuc(la[0])




