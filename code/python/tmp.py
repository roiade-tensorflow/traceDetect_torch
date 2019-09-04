
from  __future__ import  division
import torch
import  torch.nn as nn
a=torch.tensor([[1,0,0],
                [0,1,0]],dtype=torch.float)

b=torch.tensor([[1,0,0],
                [0,0,1]],dtype=torch.float)


num=1
for i in range(len(torch.tensor(a.size()) ) ):
    num*=a.size()[i]
c=torch.pow(a-b,2)
d=torch.pow((num/10)*(a-b)*a,2)
f=torch.sum(c+d)/num



print(a.size()[0]*a.size()[1])
print(c)
print(d)
print(f)
num=1
for i in range(len(torch.tensor(a.size()) ) ):
    num*=a.size()[i]
print(num)


