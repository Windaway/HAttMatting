import cv2
import numpy as np
import random
import math
import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms

#
# data_transforms = {
#     'train': transforms.Compose([
#         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ]),
#     'valid': transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#     ]),
# }

class Train_Dataset(Dataset):
    def __init__(self, split='train',fg_path='./DATAS/Train/FGo',bg_path='./DATAS/Train/BGo',mask_path='./DATAS/Train/GTo',image_path='./DATAS/Train/Imageo'):
        self.split = split
        files=os.listdir(fg_path)
        self.fg_imgs=[]
        for file in files:
            self.fg_imgs.append( os.path.join(fg_path,file)  )
        files=os.listdir(mask_path)
        self.m_imgs=[]
        for file in files:
            self.m_imgs.append( os.path.join(mask_path,file)  )
        files=os.listdir(bg_path)
        self.bg_imgs=[]
        for file in files:
            self.bg_imgs.append( os.path.join(bg_path,file)  )
        self.i_imgs=[]
        files = os.listdir(image_path)
        for file in files:
            self.i_imgs.append( os.path.join(image_path,file)  )

    def random_crop(self, img,mask,r):
        h, w, _ = img.shape
        crop_height, crop_width = r
        retimg = np.zeros((crop_height, crop_width,3), np.uint8)
        retmask = np.zeros((crop_height, crop_width), np.uint8)
        if w-r[1]-1>0:
            w_=random.randint(0,w-r[1]-1)
            pw=w
        else:
            w_=0
            pw=w
        if h-r[0]-1>0:
            h_=random.randint(0,h-r[0]-1)
            ph=h
        else:
            h_=0
            ph=h
        retimg[0:ph,0:pw] = img[h_:h_+r[0],w_:w_+r[1]]
        retmask[0:ph,0:pw]=mask[h_:h_+r[0],w_:w_+r[1]]
        return retimg,retmask


        # h,w,_=img.shape
        #
        # w_=random.randint(0,w-r[1]-1)
        # h_=random.randint(0,h-r[0]-1)
        # img=img[h_:h_+r[0],w_:w_+r[1]]
        # mask=mask[h_:h_+r[0],w_:w_+r[1]]
        # return img,mask

    def __getitem__(self, i):
        img= self.i_imgs[i]
        alpha=self.m_imgs[i]
        img=cv2.imread(img)
        mask= cv2.imread(alpha,0)
        different_sizes = [(512, 512), (640, 640), (800, 800), (960, 960)]
        crop_size = random.choice(different_sizes)
        img,mask=self.random_crop(img,mask,crop_size)
        if np.random.random_sample() > 0.5:
            img = np.fliplr(img)
            mask = np.fliplr(mask)
        img=cv2.resize(img,(512,512),interpolation=cv2.INTER_CUBIC)
        mask=cv2.resize(mask,(512,512),interpolation=cv2.INTER_CUBIC)
        img=np.array(img,np.float32)/255.
        mask=np.array(mask,np.float32)/255.
        img=np.transpose(img,[2,0,1])
        mask=mask[np.newaxis,:,:]
        return img, mask

    def __len__(self):
        return len(self.i_imgs)


class Val_Dataset(Dataset):
    def __init__(self, image_path='./DATAS/Val/'):
        self.i_imgs=[]
        self.i_imgsn=[]
        files = os.listdir(image_path)
        for file in files:
            self.i_imgs.append(os.path.join(image_path,file)  )
            self.i_imgsn.append(file)

    def __getitem__(self, i):
        img= self.i_imgs[i]
        imgn=self.i_imgsn[i]
        img=cv2.imread(img)
        img=cv2.resize(img,(512,512),interpolation=cv2.INTER_CUBIC)
        img=np.array(img,np.float32)/255.
        img=np.transpose(img,[2,0,1])
        return img, imgn

    def __len__(self):
        return len(self.i_imgs)

if __name__=='__main__':
    from torch.utils import data
    a=Val_Dataset()
    trainloader = data.DataLoader(a, batch_size=1, num_workers=2, shuffle=True)
    for x,y in enumerate(trainloader):
        d,n=y
        print(n,d.shape)