import torch
import  torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import cv2
import torchvision.models.resnet




class GN_(nn.Module):
    def __init__(self,ch,g=32):
        super(GN_, self).__init__()
        self.gn=nn.GroupNorm(g,ch)
    def forward(self,x):
        return self.gn(x)


class Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilations=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                                     padding, dilations, groups, bias)
    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                            keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        # std = (weight).view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)




def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilations=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = GN_
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        self.conv1 = conv3x3(inplanes, planes, stride,dilation=dilation)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ELU(inplace=True)
        self.conv2 = conv3x3(planes, planes,dilation=dilation)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = GN_
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ELU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, os=32,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = GN_
        self._norm_layer = norm_layer
        self.inplanes = 64
        if os == 16:
            stride_list = [1,2,2,1]
            dilations = [1, 1, 1, 2]
        elif os == 8:
            stride_list = [1,2,1,1]
            dilations = [1, 1, 2, 4]
        elif os == 32:
            stride_list = [1,2,2,2]
            dilations = [1, 1, 1, 1]
        else:
            raise ValueError('res2netgn: output stride=%d is not supported.'%os)
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ELU(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1,return_indices=True)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=stride_list[0],dilate=dilations[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=stride_list[1],dilate=dilations[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=stride_list[2],dilate=dilations[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=stride_list[3],dilate=dilations[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=1):
        norm_layer = self._norm_layer
        downsample = None
        self.dilation = dilate
        if stride != 1 or dilate!=1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, self.dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x,return_feat=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        convo=[x]
        # x,idx = self.maxpool(x)
        x = self.layer1(x)
        convo.append(x)
        x = self.layer2(x)
        convo.append(x)
        x = self.layer3(x)
        convo.append(x)
        x = self.layer4(x)
        convo.append(x)
        if return_feat:
            return convo
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def _resnet(arch, block, layers, pretrained, files,os, **kwargs):
    model = ResNet(block, layers,os=os, **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(files))
    return model

def resnet50(pretrained=False, files=None,os=8, **kwargs):
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained,files,os,
                   **kwargs)

class HAM(nn.Module):
    def __init__(self,resnet=False):
        super(HAM, self).__init__()
        if resnet:
            self.backb=resnet50(os=16)
        else:
            import model_org
            self.backb = model_org.resnext101_32x8d(pretrained=True)
        self.aspp_1=nn.Sequential(nn.AdaptiveAvgPool2d((1,1)),Conv2d(2048, 256, kernel_size=1),nn.GroupNorm(32,256),nn.ELU())
        self.aspp_2=nn.Sequential(Conv2d(2048, 256, 3,1,18,18),nn.GroupNorm(32,256),nn.ELU())
        self.aspp_3=nn.Sequential(Conv2d(2048, 256, 3,1,6,6),nn.GroupNorm(32,256),nn.ELU())
        self.aspp_4=nn.Sequential(Conv2d(2048, 256, 3,1,3,3),nn.GroupNorm(32,256),nn.ELU())
        self.aspp_5=nn.Sequential(Conv2d(2048, 256, 1,1,0,1),nn.GroupNorm(32,256),nn.ELU())
        self.aspp_o=nn.Sequential(Conv2d(1280, 256, 3,1,1),nn.GroupNorm(32,256),nn.ELU())
        self.up4x=nn.Upsample(scale_factor=(4,4),mode='bilinear',align_corners=True)
        self.maxpool=nn.AdaptiveMaxPool2d((1,1))
        self.se=nn.Sequential(nn.Conv2d(256,128,1,1,0,bias=True),nn.ReLU(True),nn.Conv2d(128,256,1,1,0,bias=True),nn.Sigmoid())
        self.hamin=nn.Sequential(Conv2d(256, 256, 3,1,1),nn.GroupNorm(32,256),nn.ELU())
        self.ham1=nn.Sequential(Conv2d(256, 256, (1,7),1,(0,3)),nn.GroupNorm(32,256),nn.ELU(),Conv2d(256, 128, (7,1),1,(3,0)),nn.GroupNorm(32,128),nn.ELU())
        self.ham2=nn.Sequential(Conv2d(256, 256, (7,1),1,(3,0)),nn.GroupNorm(32,256),nn.ELU(),Conv2d(256, 128, (1,7),1,(0,3)),nn.GroupNorm(32,128),nn.ELU())
        self.hamse = nn.Sequential(Conv2d(256, 256,1,1,0,bias=True), nn.Sigmoid())
        self.ham3x3 = nn.Sequential(Conv2d(256, 256,3,1,1,bias=False),nn.GroupNorm(32,256),nn.ELU())
        self.hamo= nn.Sequential(Conv2d(512, 1,3,1,1,bias=True))
        self.up2x=nn.Upsample(scale_factor=(2,2),mode='bilinear',align_corners=True)

    def forward(self,img):
        feats=self.backb(img,True)
        a_feat=feats[-1]
        aspp_1=self.aspp_1(a_feat)
        aspp_2=self.aspp_2(a_feat)
        aspp_3=self.aspp_3(a_feat)
        aspp_4=self.aspp_4(a_feat)
        aspp_5=self.aspp_5(a_feat)
        aspp_1=torch.zeros_like(aspp_5)+aspp_1
        aspp=torch.cat([aspp_1,aspp_2,aspp_3,aspp_4,aspp_5],1)
        aspp=self.aspp_o(aspp)
        aspp=self.up4x(aspp)
        aspp_mp=self.maxpool(aspp)
        aspp_mp=self.se(aspp_mp)
        aspp=aspp*aspp_mp
        aspp=self.hamin(aspp)
        aspp1=self.ham1(aspp)
        aspp2=self.ham2(aspp)
        asppse=torch.cat([aspp1,aspp2],1)
        asppse=self.hamse(asppse)
        asppse=asppse*feats[1]
        asppse= self.ham3x3(asppse)
        aspp=torch.cat([aspp,asppse],1)
        aspp=self.hamo(aspp)
        aspp=self.up2x(aspp)
        out=torch.sigmoid(aspp)
        return out


class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=4, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, input):
        out=self.model(input)
        return torch.mean(out,[1,2,3])


def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.randn(real_samples.size(0), 1, 1, 1)
    if torch.cuda.is_available():
        alpha = alpha.cuda()

    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size())
    if torch.cuda.is_available():
        fake = fake.cuda()

    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

if __name__=='__main__':
    # a=resnet34(True,os=32,files='1.pth')
    # a=NLayerDiscriminator()
    a=HAM()
    # torch.save(a.state_dict(),'1.pth')
    b=torch.randn(1,3,224,224)
    c=a(b)
    print(c.shape)

