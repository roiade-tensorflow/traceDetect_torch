import torch
import numpy as np
import matplotlib.pyplot as plt


def MyPlot(data):
    '''
    Draw three-dimensional three-point diagram
        input:
            cartesion array
        output:
            fig of array
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # 空间三维画图
    x = data[:, 0]  # [ 0  3  6  9 12 15 18 21]
    y = data[:, 1]  # [ 1  4  7 10 13 16 19 22]
    z = data[:, 2]  # [ 2  5  8 11 14 17 20 23]
    # 绘制散点图
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(x, y, z)
    # 添加坐标轴(顺序是Z, Y, X)
    ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
    ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
    ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
    plt.show()

def RecToPolar_3(RectData):
    '''
    Implement cartesian coordinate to polar coordinate
    imput:
        the array of cartesian coordinate
    output:
        the polar coodinate
        
    '''
#     RectData=yy
    #     print(RectData.size())
    SizeOfData=RectData.size()
    if(SizeOfData[2]==3):
        # print(RectData[0:3,:])
        ListSmall=1e-16#use a small num for illegal divition
        R=torch.norm(RectData,p=2,dim=2)+ListSmall
    #         print(R)
        Phi_Value=torch.addcdiv(torch.zeros_like(R),1,RectData[:,:,2],R)
        Phi=torch.acos(Phi_Value)#利用反余弦函数求出俯仰角
        r=torch.addcmul(torch.zeros_like(R),1,R,torch.sin(Phi))+ListSmall
        Theta_Value=torch.addcdiv(torch.zeros_like(r),1,RectData[:,:,0],r)
        SignalOfNum=torch.lt(RectData[:,:,1],torch.zeros_like(Theta_Value)).double()
        Flag_Signal_Coe=(-2*SignalOfNum+1)
        Flag_Fixed_Tail=np.pi*2*SignalOfNum
        Theta=torch.acos(Theta_Value).double()*Flag_Signal_Coe+Flag_Fixed_Tail
        result=torch.cat((torch.unsqueeze(R.double(),2),torch.unsqueeze(Theta.double(),2),torch.unsqueeze(Phi.double(),2)),dim=2)
        return(result)
def PolarToRec_3(PolarData):
    z=torch.mul(PolarData[:,:,0],torch.cos(PolarData[:,:,2]) )
    x=torch.mul(torch.mul(PolarData[:,:,0],torch.sin(PolarData[:,:,2]) ),torch.cos(PolarData[:,:,1]))
    y=torch.mul(torch.mul(PolarData[:,:,0],torch.sin(PolarData[:,:,2]) ),torch.sin(PolarData[:,:,1]))
    return(torch.cat((torch.unsqueeze(x,2),torch.unsqueeze(y,2),torch.unsqueeze(z,2)),dim=2))
def PolarDataAdd_3(tensor1,tensor2):
    '''
    design for addtion of two polar array
    input:
        two polar array
    output:
        a polar array
    '''
    Car1=PolarToRec_3(tensor1).double()
    Car2=PolarToRec_3(tensor2).double()
    return(RecToPolar_3(torch.add(Car1,Car2)))

#实现极坐标和直角坐标间的转换、极坐标加法
def RecToPolar(RectData):
    # print(RectData.type())
    defaultType = RectData.dtype
    '''
    transform array from cartesian coordinates to spherical coordinates
    input:
        x,y,z
    output:
        R,Theta,Phi

    '''
    #     RectData=local_parameter_org[:,0,:]
    SizeOfData = RectData.size()
    if (SizeOfData[1] == 3):
        # print(RectData[0:3,:])
        ListSmall = 1e-20  # use a small num for illegal divition
        ListSmall = torch.tensor(1e-20, dtype=defaultType)
        R = torch.norm(RectData, p=2, dim=1) + ListSmall
        Phi_Value = torch.addcdiv(torch.zeros_like(R), 1, RectData[:, 2], R)
        Phi_Value = torch.tensor(Phi_Value, dtype=defaultType)
        Phi = torch.acos(Phi_Value)  # 利用反余弦函数求出俯仰角
        phi = torch.tensor(Phi, dtype=defaultType)
        r = torch.addcmul(torch.zeros_like(R), 1, R, torch.sin(Phi)) + ListSmall
        r = torch.tensor(r, dtype=defaultType)
        Theta_Value = torch.addcdiv(torch.zeros_like(r), 1, RectData[:, 0], r).type_as(RectData)
        SignalOfNum = torch.lt(RectData[:, 1], torch.zeros_like(Theta_Value)).float()
        SignalOfNum = torch.tensor(SignalOfNum, dtype=defaultType)
        Flag_Signal_Coe = (-2 * SignalOfNum + 1)
        Flag_Fixed_Tail = np.pi * 2 * SignalOfNum
        Theta = torch.acos(Theta_Value) * Flag_Signal_Coe + Flag_Fixed_Tail

        return (torch.cat((R.reshape(-1, 1), Theta.reshape(-1, 1), Phi.reshape(-1, 1)), dim=1))
    elif (SizeOfData[1] == 2):
        ListSmall = 1e-20  # use a small num for illegal divition
        R = torch.norm(RectData, p=2, dim=1) + ListSmall
        Theta_Value = torch.addcdiv(torch.zeros_like(R), 1, RectData[:, 0], R).type_as(RectData)
        SignalOfNum = torch.lt(RectData[:, 1], torch.zeros_like(Theta_Value))
        Flag_Signal_Coe = (-2 * SignalOfNum + 1)
        Flag_Signal_Coe = Flag_Signal_Coe.type_as(RectData)
        Flag_Fixed_Tail = np.pi * 2 * SignalOfNum
        Flag_Fixed_Tail = Flag_Fixed_Tail.type_as(RectData)
        Theta = torch.acos(Theta_Value) * Flag_Signal_Coe + Flag_Fixed_Tail
        return (torch.cat((R.reshape(-1, 1), Theta.reshape(-1, 1)), dim=1))
    else:
        print('woring data format')
def PolarToRec(PolarData):
    '''
    Transform polar array to cartesian array
    input :
        R,Theta,Phi
    output:
        x,y,z
    
    '''
    SizeOfData=PolarData.size()
    if(SizeOfData[1]==2):
        RecData=torch.zeros_like(PolarData)
        RecData[:,0]=torch.mul(PolarData[:,0],torch.cos(PolarData[:,1]))
        RecData[:,1]=torch.mul(PolarData[:,0],torch.sin(PolarData[:,1]))
        return(RecData)
    elif(SizeOfData[1]==3):
        RecData=torch.zeros_like(PolarData)
        RecData[:,2]=torch.mul(PolarData[:,0],torch.cos(PolarData[:,2]))
        RecData[:,1]=torch.mul(torch.mul(PolarData[:,0],torch.sin(PolarData[:,2])),torch.sin(PolarData[:,1]))
        RecData[:,0]=torch.mul(torch.mul(PolarData[:,0],torch.sin(PolarData[:,2])),torch.cos(PolarData[:,1]))
        return(RecData)
    else:
        print('warning data format')
def PolarDataAdd(tensor1,tensor2):
    '''
    design for addtion of two polar array
    input:
        two polar array
    output:
        a polar array
    '''
    Car1=PolarToRec(tensor1)
    Car2=PolarToRec(tensor2)
    return(RecToPolar(torch.add(Car1,Car2)))
