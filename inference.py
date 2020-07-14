import torch
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
import time
import torch.nn as nn
import torch.optim as optim
import model
import dataset
from torch.utils import data
import loss
import cv2
save_path='./save/'
img_path='./img/'

def mkdir(dir):
    if os.path.isdir(dir):
        return 0
    else:
        os.mkdir(dir)
        return 1

if __name__=='__main__':
    netG=model.HAM()
    netG.load_state_dict(torch.load('./ckpt'))
    netG=netG.cuda()
    files=os.listdir(img_path)
    for file in files:
        imgn=os.path.join(img_path,file)
        imgn=cv2.imread(imgn)
        h,w,_= imgn.shape
        h_=(h//8+1)
        w_=(w//8+1)
        newimg=np.zeros((h_,w_,3),np.float32)
        newimg[0:h,0:w]+=imgn
        newimg=newimg/255.
        img=torch.from_numpy(newimg).cuda()
        with torch.no_grad():
            pred_mask = netG(img)
            pred=pred_mask.detach().cpu().numpy()
        pred=pred[0,0]*255.
        pred=np.clip(pred,0,255)
        pred=np.asarray(pred,np.uint8)[0:h,0:w]
        cv2.imwrite(os.path.join(save_path,file),pred)




