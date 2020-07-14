import torch
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import time
import torch.nn as nn
import torch.optim as optim
import model
import dataset
from torch.utils import data
import loss
import cv2
save_path='./ckpt'
tmp_path='./DATAS/Save/'

def mkdir(dir):
    if os.path.isdir(dir):
        return 0
    else:
        os.mkdir(dir)
        return 1

if __name__=='__main__':
    mkdir(tmp_path)
    train_dataset = dataset.Train_Dataset()
    val_dataset = dataset.Val_Dataset()
    trainloader = data.DataLoader(train_dataset, batch_size=2, num_workers=4, shuffle=True)
    valloader = data.DataLoader(val_dataset, batch_size=1, num_workers=4, shuffle=True)
    netG=model.HAM(resnet=False)
    netD=model.NLayerDiscriminator(use_sigmoid=False)
    netG=netG.cuda()
    netD=netD.cuda()
    MSE=nn.MSELoss().cuda()
    SSIM=loss.SSIM().cuda()

    optimizerG = optim.Adam(netG.parameters(), lr=0.01,weight_decay=0.0005)
    for epoch in range(10):
        netG.train()
        t_mse = 0
        t_ssim = 0
        for idx,data in enumerate(trainloader):
            img,mask=data
            img=img.cuda()
            mask=mask.cuda()
            netG.zero_grad()
            pred_mask = netG(img)
            msel=MSE(mask, pred_mask)
            ssiml=SSIM(mask, pred_mask)
            image_loss = msel+0.1*ssiml
            t_mse+=msel.item()
            t_ssim+=ssiml.item()
            if idx%100==0 and idx>0:
                print('PreTrain,Epoch,Idx,mse,ssim',epoch,idx,t_mse/100.,t_ssim/100.)
                t_mse = 0
                t_ssim = 0
            image_loss.backward()
            optimizerG.step()

    optimizerG = optim.SGD(netG.parameters(), lr=0.007,momentum=0.9,weight_decay=0.0005)
    scheduler=optim.lr_scheduler.ExponentialLR(optimizerG,0.9)
    optimizerD = optim.Adam(netD.parameters(), lr=1e-4)

    for epoch in range(100):
        netG.train()
        t_mse = 0
        t_ssim = 0
        t_ad = 0
        t_d=0
        t_g=0
        for idx,data in enumerate(trainloader):
            img,mask=data
            img=img.cuda()
            mask=mask.cuda()
            netD.zero_grad()
            pred_mask = netG(img)
            f_p=torch.cat([img,pred_mask],1)
            r_p=torch.cat([img,mask],1)
            logits_real = netD(r_p).mean()
            logits_fake = netD(f_p).mean()
            gradient_penalty =model.compute_gradient_penalty(netD,r_p,f_p)
            d_loss = logits_fake - logits_real + 10 * gradient_penalty
            t_d+=d_loss.item()
            d_loss.backward(retain_graph=True)
            optimizerD.step()
            netG.zero_grad()
            msel=MSE(mask, pred_mask)
            ssiml=SSIM(mask, pred_mask)
            image_loss = msel+0.1*ssiml
            adversarial_loss = -1 * netD(f_p).mean()
            t_ad+=adversarial_loss.item()
            g_loss = image_loss + 0.05 * adversarial_loss
            t_g+=g_loss.item()
            if idx%100==0 and idx>0:
                print('Train,Epoch,Idx,d,mse,ssim,ad,g',epoch,idx, t_d/100., t_mse/100.,t_ssim/100.,t_ad/100.,t_g/100.)
                t_mse = 0
                t_ssim = 0
            g_loss.backward()
            optimizerG.step()
        torch.save(netG.state_dict(), os.path.join(save_path,str(epoch)+'.nextckpt'))
        scheduler.step()
        netG.eval()
        mkdir(os.path.join(tmp_path,str(epoch)+'TrainNext'))
        for idx,data in enumerate(valloader):
            img,name=data
            img=img.cuda()
            with torch.no_grad():
                pred_mask = netG(img)
                pred=pred_mask.detach().cpu().numpy()
            pred=pred[0,0]*255.
            pred=np.clip(pred,0,255)
            pred=np.asarray(pred,np.uint8)
            cv2.imwrite(os.path.join(tmp_path,str(epoch)+'TrainNext',name[0]),pred)




