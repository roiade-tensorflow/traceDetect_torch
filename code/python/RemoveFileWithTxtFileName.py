import os
import os.path as osp
def remove(txtFileName):
    '''
    delete the file specified by index-txtfile
    '''
    path=osp.join('..','..','data')
    impath = osp.join('%s', 'Images', '%s')
    anapath = osp.join('%s', 'Annatations','trace', '%s')
    if not os.path.exists(osp.join(path,'Main',txtFileName)):
        print('txt file dose not exists')
        return
    txtFile=open(osp.join(path,'Main',txtFileName),'r')
    txtContents=txtFile.readlines()
    remove=[]
    for item in range(len(txtContents)):
        im=impath % (path,txtContents[item].strip() )
        ana=anapath % (path,txtContents[item].strip() )
        remove+=[item]
        if os.path.exists(im):
            os.remove(im)
        if os.path.exists(ana):
            os.remove(ana)
    for i,item in enumerate( remove):
        txtContents.remove(txtContents[item-i])
    txtFile.close()
    txtFile=open(osp.join(path,'Main','a.txt'),'w')
    for item in range(len(txtContents)):
        txtFile.write(txtContents[item])
    txtFile.close()
    # print(osp.join(path,'Main',txtFileName))
    os.remove(osp.join(path,'Main',txtFileName))
    print('remove file successfully ！\n')
if __name__=='__main__':
    remove('128_5_20.txt')