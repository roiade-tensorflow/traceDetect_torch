import torch
import numpy as np
import xlrd

class vecOpration():
    def __init__(self):
        pass
    def car2pol(self,car):
        '''
        input: num*2  x,y
        output:num*2  r,theta
        '''
        # 实现直角坐标转极坐标
        car = torch.tensor(car)
        if (car.size()[1] == 2):
            pol = torch.zeros_like(car)
            pol[:, 0:1] = torch.norm(car, dim=1, keepdim=True)
            esp = 1e-10
            # 对于y=cosx 自变量的取值范围为[-1,1]得到的是[0 ,π]内的角.
            value = car[:, 0:1] / (pol[:, 0:1] + esp)
            area1 = torch.tensor((car[:, 1:2] < 0) * 1).float()
            pol[:, 1:2] = torch.acos(value)
            pol[:, 1:2] = pol[:, 1:2] * torch.tensor(-2 * area1 + 1, dtype=pol.dtype) \
                          + torch.tensor(2 * (np.pi) * area1,dtype=pol.dtype)
            return (pol)
        if (car.size()[1] == 3):
            pol = torch.zeros_like(car)
            pol[:, 0] = torch.norm(car, dim=1)
            esp = 1e-10
            value = torch.div(car[:, 0:1], (torch.norm(car[:, 0:2], dim=1, keepdim=True) + esp))
            area1 = torch.tensor((car[:, 1:2] < 0) * 1).float()
            pol[:, 1:2] = torch.acos(value)
            pol[:, 1:2] = pol[:, 1:2] * torch.tensor(-2 * area1 + 1, dtype=pol.dtype)\
                          + torch.tensor(2 * (np.pi) * area1,dtype=pol.dtype)
            pol[:, 2:3] = torch.acos(car[:, 2:3] / (pol[:, 0:1] + esp))
            return (pol)

    def pol2car(self,polar):
        polar = torch.tensor(polar)
        car = torch.zeros_like(polar)
        if car.size()[1] == 3:
            car[:, 0:1] = polar[:, 0:1] * torch.sin(polar[:, 2:3]) * torch.cos(polar[:, 1:2])
            car[:, 1:2] = polar[:, 0:1] * torch.sin(polar[:, 2:3]) * torch.sin(polar[:, 1:2])
            car[:, 2:3] = polar[:, 0:1] * torch.cos(polar[:, 2:3])
            return car
        if car.size()[1] == 2:
            car[:, 0:1] = polar[:, 0:1] * torch.cos(polar[:, 1:2])
            car[:, 1:2] = polar[:, 0:1] * torch.sin(polar[:, 1:2])
            return car
    def excelRead(self,fileName):
        data=np.zeros(shape=(0,5))
        workbook = xlrd.open_workbook(fileName)  #(1)取得excel book对象
        s12 = workbook.sheet_by_name("Sheet1")  #(2)取得sheet对象
        rows = s12.nrows #(3)获得总行数
        for r in range(0,rows):
            row = s12.row_values(r) #(4)获取行数据
            data=np.concatenate((data,np.array(row).reshape(1,-1)),0)
        return data
    def polAdd(self,pol1,pol2):
        car1=self.pol2car(pol1)
        car2=self.pol2car(pol2)
        # car1=torch.tensor(car1)
        # car2=torch.tensor(car2)
        if car1.size()==car2.size():
            # print('car1\n',car1[0:10,:])
            result=car1+car2
            resultPlo=self.car2pol(result)
            # print('result',result[0:10,:])
            # print('resultPlo',resultPlo[:10,:])
            return resultPlo
        else:
            print('Data dimensions to be processed do not match\n')
            return None




if __name__=='__main__':
    vecOp=vecOpration()
    # data=vecOp.excelRead('E:\\学习\论文\毕业论文\\代码\\traceDetect_torch\\code\\python\\三角函数表.xlsx')
    pol=np.array([[3,11*np.pi/6]])
    # pol=vecOp.car2pol(data[:,3:5])
    add=vecOp.polAdd(pol,pol)
    print('pol:',pol)
    print('add:',add)

