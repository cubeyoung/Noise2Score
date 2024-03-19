import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040,0), rgb_std=(1.0, 1.0, 1.0,1.0), sign=-1):
        super(MeanShift, self).__init__(4, 4, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(4).view(4, 4, 1, 1) / std.view(4, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False
            
class MeanShift_2(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift_2, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False
            
class BasicBlock(nn.Sequential):
    def __init__(
        self, conv, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [conv(in_channels, out_channels, kernel_size, bias=bias)]
        if bn:
            m.append(nn.BatchNorm2d(out_channels))
        if act is not None:
            m.append(act)

        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
class PixelUnShuffle(nn.Module):
    """
    Inverse process of pytorch pixel shuffle module
    """
    def __init__(self, down_scale):
        """
        :param down_scale: int, down scale factor
        """
        super(PixelUnShuffle, self).__init__()

        if not isinstance(down_scale, int):
            raise ValueError('Down scale factor must be a integer number')
        self.down_scale = down_scale

    def forward(self, input):
        """
        :param input: tensor of shape (batch size, channels, height, width)
        :return: tensor of shape(batch size, channels * down_scale * down_scale, height / down_scale, width / down_scale)
        """
        b, c, h, w = input.size()
        assert h % self.down_scale == 0
        assert w % self.down_scale == 0

        oc = c * self.down_scale ** 2
        oh = int(h / self.down_scale)
        ow = int(w / self.down_scale)

        output_reshaped = input.reshape(b, c, oh, self.down_scale, ow, self.down_scale)
        output = output_reshaped.permute(0, 1, 3, 5, 2, 4).reshape(b, oc, oh, ow)

        return output        
class Block(nn.Module):

    def __init__(self,args,input_channels):
        super(Block, self).__init__()
        conv=default_conv      
        kernel_size = 3
        act = nn.ReLU(True)
        m_body1 = [ResBlock(conv, input_channels, kernel_size, act=act, res_scale=args.res_scale)]
        m_body2 =  [ResBlock(conv, input_channels, kernel_size, act=act, res_scale=args.res_scale)]
        m_body3 =  [ResBlock(conv, input_channels, kernel_size, act=act, res_scale=args.res_scale)]
        self.body1 = nn.Sequential(*m_body1)
        self.body2 = nn.Sequential(*m_body2)
        self.body3 = nn.Sequential(*m_body3)
        self.side1_0 = nn.Conv2d(input_channels, 32, kernel_size=1, stride=1, padding=0)
        self.side1_1 = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        self.side2_0 = nn.Conv2d(input_channels,32, kernel_size=1, stride=1, padding=0)
        self.side2_1 = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        self.side3_0 = nn.Conv2d(input_channels, 32, kernel_size=1, stride=1, padding=0)
        self.side3_1 = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)   

        self.fmap1 = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)
        #self.Upsample = Upsampler(conv,8,input_channels)
        self.n_GPUS = args.device_number
       
    def forward(self, x):
        weight_deconv =  make_bilinear_weights(16, 256).cuda(self.n_GPUS)
        x = torch.nn.functional.conv_transpose2d(x, weight_deconv, stride=8)
        img_H, img_W = x.shape[2], x.shape[3]
        x = self.body1(x)
        f1 = self.side1_0(x)
        f1 = self.side1_1(f1)
        x = self.body2(x)
        f2 = self.side2_0(x)
        f2 = self.side2_1(f2)
        x = self.body3(x)
        f3 = self.side3_0(x)
        f3 = self.side3_1(f3)
        side = self.fmap1(torch.cat([f1,f2,f3],1))
        side = torch.sigmoid(side)
        return side        
class Block1(nn.Module):

    def __init__(self,args,input_channels):
        super(Block1, self).__init__()
        conv=default_conv      
        self.conv1_0 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        self.side1_0 = nn.Conv2d(input_channels, 1, kernel_size=1, stride=1, padding=0)
        
        self.conv2_0 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        self.side2_0 = nn.Conv2d(input_channels, 1, kernel_size=1, stride=1, padding=0)

        self.conv3_0 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        self.side3_0 = nn.Conv2d(input_channels, 1, kernel_size=1, stride=1, padding=0)
        
        self.conv4_0 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        self.side4_0 = nn.Conv2d(input_channels, 1, kernel_size=1, stride=1, padding=0)       

        self.fmap1 = nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)
        self.maxpool =  nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        #self.Upsample = Upsampler(conv,8,input_channels)
        self.n_GPUS = args.device_number
       
    def forward(self, x):        
        img_H, img_W = x.shape[2], x.shape[3]
        weight_deconv =  make_bilinear_weights(16, 1).cuda(self.n_GPUS)
        x = torch.nn.functional.conv_transpose2d(x, weight_deconv2, stride=8)
        x = self.conv1_0(x)
        f1 = self.side1_0(x)
        x = self.maxpool(x)
        x = self.conv2_0(x)
        f2 = self.side2_0(x)
        x = self.maxpool(x)
        x = self.conv3_0(x)
        f3 = self.side3_0(x)
        x = self.maxpool(x)
        x = self.conv4_0(x)
        f4 = self.side4_0(x)
        weight_deconv2 =  make_bilinear_weights(4, 1).cuda(self.n_GPUS)
        f2 = torch.nn.functional.conv_transpose2d(f2, weight_deconv2, stride=2)   
        weight_deconv3 =  make_bilinear_weights(8, 1).cuda(self.n_GPUS)
        f3 = torch.nn.functional.conv_transpose2d(f3, weight_deconv3, stride=4)   
        weight_deconv4 =  make_bilinear_weights(16, 1).cuda(self.n_GPUS)
        f4 = torch.nn.functional.conv_transpose2d(f4, weight_deconv4, stride=8)
        f1 = crop(f1, img_H, img_W)
        f2 = crop(f2, img_H, img_W)
        f3 = crop(f3, img_H, img_W)
        f4 = crop(f4, img_H, img_W)
        side = self.fmap1(torch.cat([f1,f2,f3,f4],1))
        side = torch.sigmoid(side)
        return side
    
class Block2(nn.Module):

    def __init__(self,args,input_channels):
        super(Block2, self).__init__()
        conv=default_conv        
        self.conv1_0 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        self.side1_0 = nn.Conv2d(input_channels, 1, kernel_size=1, stride=1, padding=0)
        
        self.conv2_0 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        self.side2_0 = nn.Conv2d(input_channels, 1, kernel_size=1, stride=1, padding=0)

        self.conv3_0 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        self.side3_0 = nn.Conv2d(input_channels, 1, kernel_size=1, stride=1, padding=0)
     
    
        self.n_GPUS = args.device_number
        self.fmap1 = nn.Conv2d(3, 1, kernel_size=3, stride=1, padding=1)
        self.maxpool =  nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.Upsample = Upsampler(conv,8,input_channels)
    def forward(self, x):
        
        x = self.Upsample(x)
        img_H, img_W = x.shape[2], x.shape[3]
        x = self.conv1_0(x)
        f1 = self.side1_0(x)
        x = self.maxpool(x)
        x = self.conv2_0(x)
        f2 = self.side2_0(x)
        x = self.maxpool(x)
        x = self.conv3_0(x)
        f3 = self.side3_0(x)
        weight_deconv2 =  make_bilinear_weights(4, 1).cuda(self.n_GPUS)
        f2 = torch.nn.functional.conv_transpose2d(f2, weight_deconv2, stride=2)   
        weight_deconv3 =  make_bilinear_weights(8, 1).cuda(self.n_GPUS)
        f3 = torch.nn.functional.conv_transpose2d(f3, weight_deconv3, stride=4)   
        f1 = crop(f1, img_H, img_W)
        f2 = crop(f2, img_H, img_W)
        f3 = crop(f3, img_H, img_W)
        side = self.fmap1(torch.cat([f1,f2,f3],1))
        side = torch.sigmoid(side)
        return side

class Block3(nn.Module):

    def __init__(self,args,input_channels):
        super(Block3, self).__init__()
        conv=default_conv        
        self.conv1_0 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        self.side1_0 = nn.Conv2d(input_channels, 1, kernel_size=1, stride=1, padding=0)
        
        self.conv2_0 = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1)
        self.side2_0 = nn.Conv2d(input_channels, 1, kernel_size=1, stride=1, padding=0)
 
        self.n_GPUS = args.device_number
        self.fmap1 = nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)
        self.maxpool =  nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.Upsample = Upsampler(conv,8,input_channels)
    def forward(self, x):
        x = self.Upsample(x)
        img_H, img_W = x.shape[2], x.shape[3]
        x = self.conv1_0(x)
        f1 = self.side1_0(x)
        x = self.maxpool(x)
        x = self.conv2_0(x)
        f2 = self.side2_0(x)
        weight_deconv2 =  make_bilinear_weights(4, 1).cuda(self.n_GPUS)
        f2 = torch.nn.functional.conv_transpose2d(f2, weight_deconv2, stride=2)   
        f1 = crop(f1, img_H, img_W)
        f2 = crop(f2, img_H, img_W)        
        side = self.fmap1(torch.cat([f1,f2],1))
        side = torch.sigmoid(side)
        return side     
def make_bilinear_weights(size, num_channels):
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    # print(filt)
    filt = torch.from_numpy(filt)
    w = torch.zeros(num_channels, num_channels, size, size)
    w.requires_grad = False
    for i in range(num_channels):
        for j in range(num_channels):
            if i == j:
                w[i, j] = filt
    return w    
def crop(variable, th, tw):
        h, w = variable.shape[2], variable.shape[3]
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return variable[:, :, y1 : y1 + th, x1 : x1 + tw]  