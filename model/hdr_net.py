from model.encoder import *
from model.decoder import decoder
from model.qkv import *
from model.merging import Merginglayer
from model.unet import *
from torch.autograd import Variable
from model.SwinIR import *
from model.transformer import CDFormer_SR
import os
from torch.nn import Mish
from model.restormer import *

def make_model(args):
    return HDRnet_10(args)

#  beta_schedule, beta_start, beta_end, num_diffusion_timesteps, device, nfeat=64
class HDRnet(nn.Module):
    def __init__(self, args):
        super(HDRnet, self).__init__()
        from model.diffusion import diffusion
        self.args = args
        timesteps = 4
        self.nfeat = self.args.nfeat
        self.unet_l = Unet_main_time(self.nfeat)
        self.unet_h = Unet_main_time(self.nfeat)
        #self.unet_l = Unet_main(self.nfeat)
        #self.unet_h = Unet_main(self.nfeat)
        self.encoder_l = Encoder_gt(feats=64)
        self.encoder_h = Encoder_gt(feats=64)
        self.condition = Encoder_lr(feats=64)
        self.denoise = denoise(feats=64, timesteps=4)
        self.dm1 = diffusion.DDPM(denoise=self.denoise,
                                   condition=self.condition, feats=64, timesteps=timesteps)
        self.dm2 = diffusion.DDPM(denoise=self.denoise,
                                   condition=self.condition, feats=64, timesteps=timesteps)
        self.merge = Merginglayer()

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, x):
        sample = x[0]
        diffusion = x[1]
        low = sample['low_image']
        mid = sample['mid_image']
        high = sample['high_image']
        gt = sample['hdr_gt']
        expos = sample['expos']
        expos_att = sample['expos_att']
        expos_att = expos_att.float()
        if self.training:
            if not diffusion:
                ## feature map
                fea1 = self.encoder_l(mid, low)# xiang liang
                fea2 = self.encoder_h(mid, high)
                fea_diff1 = fea1
                fea_diff2 = fea2
                # mid_fea1 = self.condition1(mid)
                # mid_fea2 = self.condition2(mid)
                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                #_low = self.unet_l(mid, fea_diff1)
                #_high = self.unet_h(mid, fea_diff2)
                # _low = self.decoder1(_low)
                # _high = self.decoder2(_high)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
            else:
                self.freeze_module(self.encoder_l)
                self.freeze_module(self.encoder_h)
#                self.freeze_module(self.unet_l)
#                self.freeze_module(self.unet_h)              

      
                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)

                fea_diff1 = self.dm1(mid, fea1)
                fea_diff2 = self.dm2(mid, fea2)

                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                #_low = self.unet_l(mid, fea_diff1)
                #_high = self.unet_h(mid, fea_diff2)

                #
                # _low = self.decoder1(_low)
                # _high = self.decoder2(_high)
                #
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)

            return fea_diff1, fea_diff2, fea1, fea2, _low, _high, hdr
        else:
             
            if not diffusion:
                fea1 = self.encoder_l(mid, low)  # b,1,h,w
                fea2 = self.encoder_h(mid, high)
                #fea1 = F.softmax(fea1, dim=1)  # 沿类别维度归一化
                #fea2 = F.softmax(fea2, dim=1)  # 沿类别维度归一化
                fea_diff1 = fea1
                fea_diff2 = fea2

                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                #_low = self.unet_l(mid, fea_diff1)
                #_high = self.unet_h(mid, fea_diff2)
        
                # _low = self.decoder1(_low)
                # _high = self.decoder2(_high)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
            else:
                fea1 = self.encoder_l(mid, low) 
                fea2 = self.encoder_h(mid, high)
                #s_fea1 = F.softmax(fea1, dim=1)  # 沿类别维度归一化
#                s_fea2 = F.softmax(fea2, dim=1)  # 沿类别维度归一化
                fea_diff1 = self.dm1(mid)
                fea_diff2 = self.dm2(mid)
                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                #_low = self.unet_l(mid, fea_diff1)
                #_high = self.unet_h(mid, fea_diff2)
                #
                # _low = self.decoder1(_low)
                # _high = self.decoder2(_high)

                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
#            return fea1, fea2,fea_diff1,fea_diff2,c1,c2,hdr
            return hdr
            
# no diffusion  ,zhi jie  loss(encoder1 he encoder2)
class HDRnet_1(nn.Module):
    def __init__(self, args):
        super(HDRnet_1, self).__init__()
        self.args = args
        timesteps = 4
        self.nfeat = self.args.nfeat
        #self.unet_l = Unet_main_time(self.nfeat)
        #self.unet_h = Unet_main_time(self.nfeat)
        self.unet_l = Unet_main(self.nfeat)
        self.unet_h = Unet_main(self.nfeat)
        self.encoder_l = Encoder_gt(feats=64)
        self.encoder_h = Encoder_gt(feats=64)
        self.condition_l = Encoder_lr(feats=64)
        self.condition_h = Encoder_lr(feats=64)
#        self.denoise = denoise(feats=64, timesteps=4)
#        self.dm1 = diffusion.DDPM(denoise=self.denoise,
#                                   condition=self.condition, feats=64, timesteps=timesteps)
#        self.dm2 = diffusion.DDPM(denoise=self.denoise,
#                                   condition=self.condition, feats=64, timesteps=timesteps)
        self.merge = Merginglayer()

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, x):
        sample = x[0]
        diffusion = x[1]
        low = sample['low_image']
        mid = sample['mid_image']
        high = sample['high_image']
        gt = sample['hdr_gt']
        expos = sample['expos']
        expos_att = sample['expos_att']
        expos_att = expos_att.float()
        if self.training:
            if not diffusion:
                ## feature map
                fea1 = self.encoder_l(mid, low)# xiang liang
                fea2 = self.encoder_h(mid, high)
                fea_diff1 = fea1
                fea_diff2 = fea2
                #_low = self.unet_l(mid, fea_diff1,expos_att,True)
                #_high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
            else:
                self.freeze_module(self.encoder_l)
                self.freeze_module(self.encoder_h)              

                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)
                
#                fea_diff1 = self.dm1(mid, fea1)
#                fea_diff2 = self.dm2(mid, fea2)

                fea_diff1 = self.condition_l(mid)
                fea_diff2 = self.condition_h(mid)


#                _low = self.unet_l(mid, fea_diff1,expos_att,True)
#                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)


                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)

            return fea_diff1, fea_diff2, fea1, fea2, _low, _high, hdr
        else:  
            if not diffusion:
                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)
                #fea1 = F.softmax(fea1, dim=1)  # 沿类别维度归一化
                #fea2 = F.softmax(fea2, dim=1)  # 沿类别维度归一化
                fea_diff1 = fea1
                fea_diff2 = fea2

#                _low = self.unet_l(mid, fea_diff1,expos_att,True)
#                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)
        
                # _low = self.decoder1(_low)
                # _high = self.decoder2(_high)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
            else:
                fea1 = self.encoder_l(mid, low) 
                fea2 = self.encoder_h(mid, high)
                #s_fea1 = F.softmax(fea1, dim=1)  # 沿类别维度归一化
#                s_fea2 = F.softmax(fea2, dim=1)  # 沿类别维度归一化
                fea_diff1 = self.condition_l(mid)
                fea_diff2 = self.condition_h(mid)
#                _low = self.unet_l(mid, fea_diff1,expos_att,True)
#                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)


                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
#            return fea1, fea2,fea_diff1,fea_diff2,hdr
            return hdr
            
class HDRnet_2(nn.Module):
    def __init__(self, args):
        super(HDRnet_2, self).__init__()
        self.args = args
        timesteps = 4
        self.nfeat = self.args.nfeat
        #self.unet_l = Unet_main_time(self.nfeat)
        #self.unet_h = Unet_main_time(self.nfeat)
        self.unet_l = Unet_main(self.nfeat)
        self.unet_h = Unet_main(self.nfeat)
        self.encoder_l = Encoder_gt(feats=64)
        self.encoder_h = Encoder_gt(feats=64)
        self.condition_l = Encoder_lr(feats=64)
        self.condition_h = Encoder_lr(feats=64)
        self.head_l=MLP()
        self.head_h=MLP()
#        self.denoise = denoise(feats=64, timesteps=4)
#        self.dm1 = diffusion.DDPM(denoise=self.denoise,
#                                   condition=self.condition, feats=64, timesteps=timesteps)
#        self.dm2 = diffusion.DDPM(denoise=self.denoise,
#                                   condition=self.condition, feats=64, timesteps=timesteps)
        self.merge = Merginglayer()

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, x):
        sample = x[0]
        diffusion = x[1]
        low = sample['low_image']
        mid = sample['mid_image']
        high = sample['high_image']
        gt = sample['hdr_gt']
        expos = sample['expos']
        expos_att = sample['expos_att']
        expos_att = expos_att.float()
        if self.training:
            if not diffusion:
                ## feature map
                fea1 = self.encoder_l(mid, low)# xiang liang
                fea2 = self.encoder_h(mid, high)
                fea_diff1 = fea1
                fea_diff2 = fea2
                #_low = self.unet_l(mid, fea_diff1,expos_att,True)
                #_high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
                l_f2 = None
                h_f2 = None
            else:
                self.freeze_module(self.encoder_l)
                self.freeze_module(self.encoder_h)              

                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)
                
                fea1 = self.head_l(fea1)
                fea2 = self.head_h(fea2)
#                fea_diff1 = self.dm1(mid, fea1)
#                fea_diff2 = self.dm2(mid, fea2)

                fea_diff1 = self.condition_l(mid)
                fea_diff2 = self.condition_h(mid)
                l_f2 = self.head_l(fea_diff1)
                h_f2 = self.head_h(fea_diff2)

#                _low = self.unet_l(mid, fea_diff1,expos_att,True)
#                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)


                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)

            return fea_diff1, fea_diff2, fea1, fea2, _low, _high, hdr,l_f2,h_f2
        else:  
            if not diffusion:
                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)
                #fea1 = F.softmax(fea1, dim=1)  # 沿类别维度归一化
                #fea2 = F.softmax(fea2, dim=1)  # 沿类别维度归一化
                fea_diff1 = fea1
                fea_diff2 = fea2

#                _low = self.unet_l(mid, fea_diff1,expos_att,True)
#                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)
        
                # _low = self.decoder1(_low)
                # _high = self.decoder2(_high)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
            else:
                fea1 = self.encoder_l(mid, low) 
                fea2 = self.encoder_h(mid, high)
                #s_fea1 = F.softmax(fea1, dim=1)  # 沿类别维度归一化
#                s_fea2 = F.softmax(fea2, dim=1)  # 沿类别维度归一化
                fea_diff1 = self.condition_l(mid)
                fea_diff2 = self.condition_h(mid)
#                _low = self.unet_l(mid, fea_diff1,expos_att,True)
#                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)


                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
#            return fea1, fea2,fea_diff1,fea_diff2,hdr
            return hdr
### mlp+time         
class HDRnet_3(nn.Module):
    def __init__(self, args):
        super(HDRnet_3, self).__init__()
        self.args = args
        timesteps = 4
        self.nfeat = self.args.nfeat
        #self.unet_l = Unet_main_time(self.nfeat)
        #self.unet_h = Unet_main_time(self.nfeat)
        self.unet_l = Unet_main(self.nfeat)
        self.unet_h = Unet_main(self.nfeat)
        self.encoder_l = Encoder_gt(feats=64)
        self.encoder_h = Encoder_gt(feats=64)
        self.condition_l = Encoder_lr(feats=64)
        self.condition_h = Encoder_lr(feats=64)
        self.head_l=MLP_time()
        self.head_h=MLP_time()
        self.merge = Merginglayer()

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, x):
        sample = x[0]
        diffusion = x[1]
        low = sample['low_image']
        mid = sample['mid_image']
        high = sample['high_image']
        gt = sample['hdr_gt']
        expos = sample['expos']
        expos_att = sample['expos_att']
        expos_att = expos_att.float()
        if self.training:
            if not diffusion:
                ## feature map
                fea1 = self.encoder_l(mid, low)# xiang liang
                fea2 = self.encoder_h(mid, high)
                fea_diff1 = fea1
                fea_diff2 = fea2
                #_low = self.unet_l(mid, fea_diff1,expos_att,True)
                #_high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
                l_f2 = None
                h_f2 = None
            else:
                self.freeze_module(self.encoder_l)
                self.freeze_module(self.encoder_h)              

                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)
                
                fea1 = self.head_l(fea1,expos_att,True)
                fea2 = self.head_h(fea2,expos_att,False)

                fea_diff1 = self.condition_l(mid)
                fea_diff2 = self.condition_h(mid)
                l_f2 = self.head_l(fea_diff1,expos_att,True)
                h_f2 = self.head_h(fea_diff2,expos_att,False)

#                _low = self.unet_l(mid, fea_diff1,expos_att,True)
#                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)


                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)

            return fea_diff1, fea_diff2, fea1, fea2, _low, _high, hdr,l_f2,h_f2
        else:  
            if not diffusion:
                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)
                #fea1 = F.softmax(fea1, dim=1)  # 沿类别维度归一化
                #fea2 = F.softmax(fea2, dim=1)  # 沿类别维度归一化
                fea_diff1 = fea1
                fea_diff2 = fea2

#                _low = self.unet_l(mid, fea_diff1,expos_att,True)
#                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)
        
                # _low = self.decoder1(_low)
                # _high = self.decoder2(_high)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
            else:
                fea1 = self.encoder_l(mid, low) 
                fea2 = self.encoder_h(mid, high)

                fea_diff1 = self.condition_l(mid)
                fea_diff2 = self.condition_h(mid)
#                _low = self.unet_l(mid, fea_diff1,expos_att,True)
#                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)


                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
#            return fea1, fea2,fea_diff1,fea_diff2,hdr
            return hdr

### mlp+time2         
class HDRnet_4(nn.Module):
    def __init__(self, args):
        super(HDRnet_4, self).__init__()
        self.args = args
        timesteps = 4
        self.nfeat = self.args.nfeat
        #self.unet_l = Unet_main_time(self.nfeat)
        #self.unet_h = Unet_main_time(self.nfeat)
        self.unet_l = Unet_main(self.nfeat)
        self.unet_h = Unet_main(self.nfeat)
        self.encoder_l = Encoder_gt(feats=64)
        self.encoder_h = Encoder_gt(feats=64)
        self.condition_l = Encoder_lr(feats=64)
        self.condition_h = Encoder_lr(feats=64)
        self.head_l=MLP_time2()
        self.head_h=MLP_time2()
        self.merge = Merginglayer()

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, x):
        sample = x[0]
        diffusion = x[1]
        low = sample['low_image']
        mid = sample['mid_image']
        high = sample['high_image']
        gt = sample['hdr_gt']
        expos = sample['expos']
        expos_att = sample['expos_att']
        expos_att = expos_att.float()
        if self.training:
            if not diffusion:
                ## feature map
                fea1 = self.encoder_l(mid, low)# xiang liang
                fea2 = self.encoder_h(mid, high)
                fea_diff1 = fea1
                fea_diff2 = fea2
                #_low = self.unet_l(mid, fea_diff1,expos_att,True)
                #_high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
                l_f2 = None
                h_f2 = None
            else:
                self.freeze_module(self.encoder_l)
                self.freeze_module(self.encoder_h)              

                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)
                
                fea1 = self.head_l(fea1,expos_att,True)
                fea2 = self.head_h(fea2,expos_att,False)

                fea_diff1 = self.condition_l(mid)
                fea_diff2 = self.condition_h(mid)
                l_f2 = self.head_l(fea_diff1,expos_att,True)
                h_f2 = self.head_h(fea_diff2,expos_att,False)

#                _low = self.unet_l(mid, fea_diff1,expos_att,True)
#                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)


                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)

            return fea_diff1, fea_diff2, fea1, fea2, _low, _high, hdr,l_f2,h_f2
        else:  
            if not diffusion:
                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)
                #fea1 = F.softmax(fea1, dim=1)  # 沿类别维度归一化
                #fea2 = F.softmax(fea2, dim=1)  # 沿类别维度归一化
                fea_diff1 = fea1
                fea_diff2 = fea2

#                _low = self.unet_l(mid, fea_diff1,expos_att,True)
#                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)
        
                # _low = self.decoder1(_low)
                # _high = self.decoder2(_high)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
            else:
                fea1 = self.encoder_l(mid, low) 
                fea2 = self.encoder_h(mid, high)

                fea_diff1 = self.condition_l(mid)
                fea_diff2 = self.condition_h(mid)
#                _low = self.unet_l(mid, fea_diff1,expos_att,True)
#                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)


                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
#            return fea1, fea2,fea_diff1,fea_diff2,hdr
            return hdr 
####only add mid mlp ，donnot add (mid+low) mlp ，loss between (mlp and encoder)
class HDRnet_5(nn.Module):
    def __init__(self, args):
        super(HDRnet_5, self).__init__()
        self.args = args
        timesteps = 4
        self.nfeat = self.args.nfeat
        #self.unet_l = Unet_main_time(self.nfeat)
        #self.unet_h = Unet_main_time(self.nfeat)
        self.unet_l = Unet_main(self.nfeat)
        self.unet_h = Unet_main(self.nfeat)
        self.encoder_l = Encoder_gt(feats=64)
        self.encoder_h = Encoder_gt(feats=64)
        self.condition_l = Encoder_lr(feats=64)
        self.condition_h = Encoder_lr(feats=64)
        self.head_l=MLP()
        self.head_h=MLP()
        self.merge = Merginglayer()

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, x):
        sample = x[0]
        diffusion = x[1]
        low = sample['low_image']
        mid = sample['mid_image']
        high = sample['high_image']
        gt = sample['hdr_gt']
        expos = sample['expos']
        expos_att = sample['expos_att']
        expos_att = expos_att.float()
        if self.training:
            if not diffusion:
                ## feature map
                fea1 = self.encoder_l(mid, low)# xiang liang
                fea2 = self.encoder_h(mid, high)
                fea_diff1 = fea1
                fea_diff2 = fea2
                #_low = self.unet_l(mid, fea_diff1,expos_att,True)
                #_high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
                l_f2 = None
                h_f2 = None
            else:
                self.freeze_module(self.encoder_l)
                self.freeze_module(self.encoder_h)              

                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)
                
                fea_diff1 = self.condition_l(mid)
                fea_diff2 = self.condition_h(mid)
                l_f2 = self.head_l(fea_diff1)
                h_f2 = self.head_h(fea_diff2)

#                _low = self.unet_l(mid, fea_diff1,expos_att,True)
#                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(mid, l_f2)
                _high = self.unet_h(mid, h_f2)


                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)

            return fea_diff1, fea_diff2, fea1, fea2, _low, _high, hdr,l_f2,h_f2
        else:  
            if not diffusion:
                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)
                #fea1 = F.softmax(fea1, dim=1)  # 沿类别维度归一化
                #fea2 = F.softmax(fea2, dim=1)  # 沿类别维度归一化
                fea_diff1 = fea1
                fea_diff2 = fea2

#                _low = self.unet_l(mid, fea_diff1,expos_att,True)
#                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)
        
                # _low = self.decoder1(_low)
                # _high = self.decoder2(_high)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
            else:
                fea_diff1 = self.condition_l(mid)
                fea_diff2 = self.condition_h(mid)
                l_f2 = self.head_l(fea_diff1)
                h_f2 = self.head_h(fea_diff2)
#                _low = self.unet_l(mid, fea_diff1,expos_att,True)
#                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(mid, l_f2)
                _high = self.unet_h(mid, h_f2)


                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
#            return fea1, fea2,fea_diff1,fea_diff2,hdr
            return hdr 

### transformer+time  xiaorong model2        
class HDRnet_6(nn.Module):
    def __init__(self, args):
        super(HDRnet_6, self).__init__()
        self.args = args
        timesteps = 4
        self.nfeat = self.args.nfeat
        #self.unet_l = Unet_main_time(self.nfeat)
        #self.unet_h = Unet_main_time(self.nfeat)
        self.unet_l = Unet_main(self.nfeat)
        self.unet_h = Unet_main(self.nfeat)
        self.encoder_l = Encoder_gt(feats=64)
        self.encoder_h = Encoder_gt(feats=64)
        self.condition_l = Encoder_lr(feats=64)
        self.condition_h = Encoder_lr(feats=64)
        self.head_l=Enhance4(256)
        self.head_h=Enhance4(256)
        self.merge = Merginglayer()

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, x):
        sample = x[0]
        diffusion = x[1]
        low = sample['low_image']
        mid = sample['mid_image']
        high = sample['high_image']
        gt = sample['hdr_gt']
        h, w = mid.shape[2], mid.shape[3]
        h = h - h % 16
        w = w - w % 16
        low = low[:, :, :h, :w]
        mid = mid[:, :, :h, :w]
        high = high[:, :, :h, :w]
        gt = gt[:, :, :h, :w]
        expos = sample['expos']
        expos_att = sample['expos_att']
        expos_att = expos_att.float()
        if self.training:
            if not diffusion:
                ## feature map
                fea1 = self.encoder_l(mid, low)# xiang liang
                fea2 = self.encoder_h(mid, high)
                fea_diff1 = fea1
                fea_diff2 = fea2
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
                l_f2 = None
                h_f2 = None
            else:
                self.freeze_module(self.encoder_l)
                self.freeze_module(self.encoder_h)              
                # GT +mid
                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)
                
                fea1 = self.head_l(fea1,expos_att,True)
                fea2 = self.head_h(fea2,expos_att,False)
                #mid
                fea_diff1 = self.condition_l(mid)
                fea_diff2 = self.condition_h(mid)
                l_f2 = self.head_l(fea_diff1,expos_att,True)
                h_f2 = self.head_h(fea_diff2,expos_att,False)

                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)

                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)

            return fea_diff1, fea_diff2, fea1, fea2, _low, _high, hdr,l_f2,h_f2
        else:  
            if not diffusion:
                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)

                fea_diff1 = fea1
                fea_diff2 = fea2
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)
        
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
            else:
                fea_diff1 = self.condition_l(mid)
                fea_diff2 = self.condition_h(mid)
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
#            return fea1, fea2,fea_diff1,fea_diff2,hdr
            return hdr,_low,_high  

### transformer_notime        
class HDRnet_7(nn.Module):
    def __init__(self, args):
        super(HDRnet_7, self).__init__()
        self.args = args
        timesteps = 4
        self.nfeat = self.args.nfeat
        #self.unet_l = Unet_main_time(self.nfeat)
        #self.unet_h = Unet_main_time(self.nfeat)
        self.unet_l = Unet_main(self.nfeat)
        self.unet_h = Unet_main(self.nfeat)
        self.encoder_l = Encoder_gt(feats=64)
        self.encoder_h = Encoder_gt(feats=64)
        self.condition_l = Encoder_lr(feats=64)
        self.condition_h = Encoder_lr(feats=64)
        self.head_l=Enhance1(256)
        self.head_h=Enhance1(256)
        self.merge = Merginglayer()

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, x):
        sample = x[0]
        diffusion = x[1]
        low = sample['low_image']
        mid = sample['mid_image']
        high = sample['high_image']
        gt = sample['hdr_gt']
        expos = sample['expos']
        expos_att = sample['expos_att']
        expos_att = expos_att.float()
        if self.training:
            if not diffusion:
                ## feature map
                fea1 = self.encoder_l(mid, low)# xiang liang
                fea2 = self.encoder_h(mid, high)
                fea_diff1 = fea1
                fea_diff2 = fea2
                #_low = self.unet_l(mid, fea_diff1,expos_att,True)
                #_high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
                l_f2 = None
                h_f2 = None
            else:
                self.freeze_module(self.encoder_l)
                self.freeze_module(self.encoder_h)              
                # GT +mid
                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)
                
                fea1 = self.head_l(fea1,expos_att,True)
                fea2 = self.head_h(fea2,expos_att,False)
                #mid
                fea_diff1 = self.condition_l(mid)
                fea_diff2 = self.condition_h(mid)
                l_f2 = self.head_l(fea_diff1,expos_att,True)
                h_f2 = self.head_h(fea_diff2,expos_att,False)

#                _low = self.unet_l(mid, fea_diff1,expos_att,True)
#                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)

                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)

            return fea_diff1, fea_diff2, fea1, fea2, _low, _high, hdr,l_f2,h_f2
        else:  
            if not diffusion:
                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)

                fea_diff1 = fea1
                fea_diff2 = fea2

#                _low = self.unet_l(mid, fea_diff1,expos_att,True)
#                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)
        
                # _low = self.decoder1(_low)
                # _high = self.decoder2(_high)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
            else:
                fea1 = self.encoder_l(mid, low) 
                fea2 = self.encoder_h(mid, high)

                fea_diff1 = self.condition_l(mid)
                fea_diff2 = self.condition_h(mid)
#                _low = self.unet_l(mid, fea_diff1,expos_att,True)
#                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)

                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
#            return fea1, fea2,fea_diff1,fea_diff2,hdr
            return hdr  
# tf_time,unet_time
class HDRnet_8(nn.Module):
    def __init__(self, args):
        super(HDRnet_8, self).__init__()
        self.args = args
        timesteps = 4
        self.nfeat = self.args.nfeat
        self.unet_l = Unet_main_time(self.nfeat)
        self.unet_h = Unet_main_time(self.nfeat)  
        self.encoder_l = Encoder_gt(feats=64)
        self.encoder_h = Encoder_gt(feats=64)
        self.condition_l = Encoder_lr(feats=64)
        self.condition_h = Encoder_lr(feats=64)
        self.head_l=Enhance4(256)
        self.head_h=Enhance4(256)
        self.merge = Merginglayer()

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, x):
        sample = x[0]
        diffusion = x[1]
        low = sample['low_image']
        mid = sample['mid_image']
        high = sample['high_image']
        gt = sample['hdr_gt']
        h, w = mid.shape[2], mid.shape[3]
        h = h - h % 16
        w = w - w % 16
        low = low[:, :, :h, :w]
        mid = mid[:, :, :h, :w]
        high = high[:, :, :h, :w]
        gt = gt[:, :, :h, :w]
        expos = sample['expos']
        expos_att = sample['expos_att']
        expos_att = expos_att.float()
        if self.training:
            if not diffusion:
                ## xianyan xinxi
                fea1 = self.encoder_l(mid, low)# xiang liang
                fea2 = self.encoder_h(mid, high)
                fea_diff1 = fea1
                fea_diff2 = fea2
                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
                l_f2 = None
                h_f2 = None
            else:
                self.freeze_module(self.encoder_l)
                self.freeze_module(self.encoder_h)              
                # GT +mid
                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)
                
                fea1 = self.head_l(fea1,expos_att,True)
                fea2 = self.head_h(fea2,expos_att,False)
                #mid
                fea_diff1 = self.condition_l(mid)
                fea_diff2 = self.condition_h(mid)
                l_f2 = self.head_l(fea_diff1,expos_att,True)
                h_f2 = self.head_h(fea_diff2,expos_att,False)

                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)

                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)

            return fea_diff1, fea_diff2, fea1, fea2, _low, _high, hdr,l_f2,h_f2
        else:  
            if not diffusion:
                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)

                fea_diff1 = fea1
                fea_diff2 = fea2

                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)

                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
            else:
                fea1 = self.encoder_l(mid, low) 
                fea2 = self.encoder_h(mid, high)

                fea_diff1 = self.condition_l(mid)
                fea_diff2 = self.condition_h(mid)
                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
#            return fea1, fea2,fea_diff1,fea_diff2,hdr
            return hdr  
### transformer+time_hf         
class HDRnet_9(nn.Module):
    def __init__(self, args):
        super(HDRnet_9, self).__init__()
        self.args = args
        timesteps = 4
        self.nfeat = self.args.nfeat
        #self.unet_l = Unet_main_time(self.nfeat)
        #self.unet_h = Unet_main_time(self.nfeat)
        self.unet_l = Unet_main(self.nfeat)
        self.unet_h = Unet_main(self.nfeat)
        self.encoder_l = Encoder_gt(feats=64)
        self.encoder_h = Encoder_gt(feats=64)
        self.condition_l = Encoder_lr(feats=64)
        self.condition_h = Encoder_lr(feats=64)
        self.head_l=Enhance4(256)
        self.head_h=Enhance4(256)
        self.merge = Merginglayer()
        self.pre_conv = nn.Sequential(nn.Conv2d(6,16, kernel_size=5, padding=2),
                                      nn.BatchNorm2d(16),
                                      Mish(),
                                      nn.Conv2d(16, 32, kernel_size=5, padding=2),
                                      nn.BatchNorm2d(32),
                                      Mish())

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, x):
        sample = x[0]
        diffusion = x[1]
        low = sample['low_image']
        mid = sample['mid_image']
        high = sample['high_image']
        gt = sample['hdr_gt']
        h, w = mid.shape[2], mid.shape[3]
        h = h - h % 16
        w = w - w % 16
        low = low[:, :, :h, :w]
        mid = mid[:, :, :h, :w]
        high = high[:, :, :h, :w]
        gt = gt[:, :, :h, :w]
        expos = sample['expos']
        expos_att = sample['expos_att']
        expos_att = expos_att.float()
        if self.training:
            if not diffusion:
                ## feature map
                midl = self.pre_conv(mid)
                fea1 = self.encoder_l(mid, low)# xiang liang
                fea2 = self.encoder_h(mid, high)
                fea_diff1 = fea1
                fea_diff2 = fea2
                #_low = self.unet_l(mid, fea_diff1,expos_att,True)
                #_high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(midl, fea_diff1)
                _high = self.unet_h(midl, fea_diff2)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
                l_f2 = None
                h_f2 = None
            else:
                self.freeze_module(self.encoder_l)
                self.freeze_module(self.encoder_h)              
                # GT +mid
                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)
                midl = self.pre_conv(mid)
                fea1 = self.head_l(fea1,expos_att,True)
                fea2 = self.head_h(fea2,expos_att,False)
                #mid
                fea_diff1 = self.condition_l(mid)
                fea_diff2 = self.condition_h(mid)
                l_f2 = self.head_l(fea_diff1,expos_att,True)
                h_f2 = self.head_h(fea_diff2,expos_att,False)

#                _low = self.unet_l(mid, fea_diff1,expos_att,True)
#                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(midl, fea_diff1)
                _high = self.unet_h(midl, fea_diff2)

                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)

            return fea_diff1, fea_diff2, fea1, fea2, _low, _high, hdr,l_f2,h_f2
        else:  
            if not diffusion:
                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)
                midl = self.pre_conv(mid)
                fea_diff1 = fea1
                fea_diff2 = fea2

#                _low = self.unet_l(mid, fea_diff1,expos_att,True)
#                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(midl, fea_diff1)
                _high = self.unet_h(midl, fea_diff2)
        
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
            else:
                fea1 = self.encoder_l(mid, low) 
                fea2 = self.encoder_h(mid, high)
                midl = self.pre_conv(mid)
                fea_diff1 = self.condition_l(mid)
                fea_diff2 = self.condition_h(mid)
#                _low = self.unet_l(mid, fea_diff1,expos_att,True)
#                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(midl, fea_diff1)
                _high = self.unet_h(midl, fea_diff2)

                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
#            return fea1, fea2,fea_diff1,fea_diff2,hdr
            return hdr  
#### transformer+time  and unet+time
class HDRnet_10(nn.Module):
    def __init__(self, args):
        super(HDRnet_10, self).__init__()
        self.args = args
        timesteps = 4
        self.nfeat = self.args.nfeat
        self.unet_l = Unet_main_time(self.nfeat)
        self.unet_h = Unet_main_time(self.nfeat)
        self.encoder_l = Encoder_gt(feats=64)
        self.encoder_h = Encoder_gt(feats=64)
        self.condition_l = Encoder_lr(feats=64)
        self.condition_h = Encoder_lr(feats=64)
        self.head_l=Enhance4(256)
        self.head_h=Enhance4(256)
        self.merge = Merginglayer()
        self.weights_initialized = True
        self.should_reset_conditions =False 

    def reset_condition_weights(self):
        if self.should_reset_conditions:
            print("Resetting condition_l and condition_h weights")
            self.condition_l.reset_parameters()
            self.condition_h.reset_parameters()
            self.weights_initialized = True
    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, x):
        #if not self.weights_initialized and self.should_reset_conditions:
        #    self.reset_condition_weights()
        sample = x[0]
        diffusion = x[1]
        low = sample['low_image']
        mid = sample['mid_image']
        high = sample['high_image']
        h, w = mid.shape[2], mid.shape[3]
        h = h - h % 16
        w = w - w % 16
        low = low[:, :, :h, :w]
        mid = mid[:, :, :h, :w]
        high = high[:, :, :h, :w]
        expos = sample['expos']
        expos_att = sample['expos_att']
        expos_att = expos_att.float()
        if self.training:
            if not diffusion:
                ## feature map
                fea1 = self.encoder_l(mid, low)# xiang liang
                fea2 = self.encoder_h(mid, high)
                fea_diff1 = fea1
                fea_diff2 = fea2
                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
                l_f2 = None
                h_f2 = None
            else:
                self.freeze_module(self.encoder_l)
                self.freeze_module(self.encoder_h)              
                # GT +mid
                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)
                fea1 = self.head_l(fea1,expos_att,True)
                fea2 = self.head_h(fea2,expos_att,False)
                #mid
                fea_diff1 = self.condition_l(mid)
                fea_diff2 = self.condition_h(mid)
                l_f2 = self.head_l(fea_diff1,expos_att,True)
                h_f2 = self.head_h(fea_diff2,expos_att,False)

                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)
               
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)

            return fea_diff1, fea_diff2, fea1, fea2, _low, _high, hdr,l_f2,h_f2
        else:  
            if not diffusion:
                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)
                fea_diff1 = fea1
                fea_diff2 = fea2

                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)
        
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
            else:
                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)

                fea_diff1 = self.condition_l(mid)
                fea_diff2 = self.condition_h(mid)

                _low = self.unet_l(mid, fea_diff1 ,expos_att,True)
                _high = self.unet_h(mid, fea_diff2 ,expos_att,False)

                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
            return hdr,_low,_high,fea_diff1, fea_diff2, fea1, fea2
            #return hdr,_low,_high

class HDRnet_11(nn.Module):
    def __init__(self, args):
        super(HDRnet_11, self).__init__()
        self.args = args
        timesteps = 4
        self.nfeat = self.args.nfeat
        self.unet_l = main_net(self.nfeat)
        self.unet_h = main_net(self.nfeat)
        self.encoder_l = Encoder_gt(feats=64)
        self.encoder_h = Encoder_gt(feats=64)
        self.condition_l = Encoder_lr(feats=64)
        self.condition_h = Encoder_lr(feats=64)
        self.head_l=Enhance4(256)
        self.head_h=Enhance4(256)
        self.merge = Merginglayer()

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, x):
        sample = x[0]
        diffusion = x[1]
        low = sample['low_image']
        mid = sample['mid_image']
        high = sample['high_image']
        gt = sample['hdr_gt']
        h, w = mid.shape[2], mid.shape[3]
        h = h - h % 16
        w = w - w % 16
        low = low[:, :, :h, :w]
        mid = mid[:, :, :h, :w]
        high = high[:, :, :h, :w]
        gt = gt[:, :, :h, :w]
        expos = sample['expos']
        expos_att = sample['expos_att']
        expos_att = expos_att.float()
        if self.training:
            if not diffusion:
                ## feature map
                fea1 = self.encoder_l(mid, low)# xiang liang
                fea2 = self.encoder_h(mid, high)
                fea_diff1 = fea1
                fea_diff2 = fea2
                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
                l_f2 = None
                h_f2 = None
            else:
                self.freeze_module(self.encoder_l)
                self.freeze_module(self.encoder_h)              
                # GT +mid
                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)
                fea1 = self.head_l(fea1,expos_att,True)
                fea2 = self.head_h(fea2,expos_att,False)
                #mid
                fea_diff1 = self.condition_l(mid)
                fea_diff2 = self.condition_h(mid)
                l_f2 = self.head_l(fea_diff1,expos_att,True)
                h_f2 = self.head_h(fea_diff2,expos_att,False)

                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)
               
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)

            return fea_diff1, fea_diff2, fea1, fea2, _low, _high, hdr,l_f2,h_f2
        else:  
            if not diffusion:
                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)
                fea_diff1 = fea1
                fea_diff2 = fea2

                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)
        
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
            else:
                fea_diff1 = self.condition_l(mid)
                fea_diff2 = self.condition_h(mid)
                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)

                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
#            return fea1, fea2,fea_diff1,fea_diff2,hdr
            return hdr 
##no tpae
class HDRnet_12(nn.Module):
    def __init__(self, args):
        super(HDRnet_12, self).__init__()
        self.args = args
        self.nfeat = self.args.nfeat
        self.unet_l = Unet_main_time(self.nfeat)
        self.unet_h = Unet_main_time(self.nfeat)
        self.encoder_l = Encoder_gt(feats=64)
        self.encoder_h = Encoder_gt(feats=64)
        self.condition_l = Encoder_lr(feats=64)
        self.condition_h = Encoder_lr(feats=64)

        self.merge = Merginglayer()

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, x):
        sample = x[0]
        diffusion = x[1]
        low = sample['low_image']
        mid = sample['mid_image']
        high = sample['high_image']
        gt = sample['hdr_gt']
        h, w = mid.shape[2], mid.shape[3]
        h = h - h % 16
        w = w - w % 16
        low = low[:, :, :h, :w]
        mid = mid[:, :, :h, :w]
        high = high[:, :, :h, :w]
        gt = gt[:, :, :h, :w]
        expos = sample['expos']
        expos_att = sample['expos_att']
        expos_att = expos_att.float()
        if self.training:
            if not diffusion:
                ## feature map
                fea1 = self.encoder_l(mid, low)# xiang liang
                fea2 = self.encoder_h(mid, high)
                fea_diff1 = fea1
                fea_diff2 = fea2
                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
                l_f2 = None
                h_f2 = None
            else:
                self.freeze_module(self.encoder_l)
                self.freeze_module(self.encoder_h)              
                # GT +mid
                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)

                #mid
                fea_diff1 = self.condition_l(mid)
                fea_diff2 = self.condition_h(mid)

                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)
               
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)

            return fea_diff1, fea_diff2, fea1, fea2, _low, _high, hdr
        else:  
            if not diffusion:
                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)
                fea_diff1 = fea1
                fea_diff2 = fea2

                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)
        
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
            else:
                fea1 = self.encoder_l(mid, low) 
                fea2 = self.encoder_h(mid, high)

                fea_diff1 = self.condition_l(mid)
                fea_diff2 = self.condition_h(mid)
                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)

                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
        #    return fea_diff1, fea_diff2, fea1, fea2,hdr
            return hdr  

#kuosan wu share weight
class HDRnet_13(nn.Module):
    def __init__(self, args):
        super(HDRnet_13, self).__init__()
        from model.diffusion import diffusion
        self.args = args
        timesteps = 4
        self.nfeat = self.args.nfeat
        self.unet_l = Unet_main_time(self.nfeat)
        self.unet_h = Unet_main_time(self.nfeat)
        self.encoder_l = Encoder_gt(feats=64)
        self.encoder_h = Encoder_gt(feats=64)
        self.condition1 = Encoder_lr(feats=64)
        self.condition2 = Encoder_lr(feats=64)
        self.denoise1 = denoise(feats=64, timesteps=4)
        self.denoise2 = denoise(feats=64, timesteps=4)
        self.dm1 = diffusion.DDPM(denoise=self.denoise1,
                                   condition=self.condition1, feats=64, timesteps=timesteps)
        self.dm2 = diffusion.DDPM(denoise=self.denoise2,
                                   condition=self.condition2, feats=64, timesteps=timesteps)
        self.merge = Merginglayer()

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, x):
        sample = x[0]
        diffusion = x[1]
        low = sample['low_image']
        mid = sample['mid_image']
        high = sample['high_image']
        gt = sample['hdr_gt']
        expos = sample['expos']
        expos_att = sample['expos_att']
        expos_att = expos_att.float()
        if self.training:
            if not diffusion:
                ## feature map
                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)
                fea_diff1 = fea1
                fea_diff2 = fea2
                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
            else:
                self.freeze_module(self.encoder_l)
                self.freeze_module(self.encoder_h)            
      
                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)
                fea_diff1 = self.dm1(mid, fea1)
                fea_diff2 = self.dm2(mid, fea2)

                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)

            return fea_diff1, fea_diff2, fea1, fea2, _low, _high, hdr
        else:  
            if not diffusion:
                fea1 = self.encoder_l(mid, low)  # b,1,h,w
                fea2 = self.encoder_h(mid, high)
                fea_diff1 = fea1
                fea_diff2 = fea2

                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
            else:

                fea_diff1 = self.dm1(mid)
                fea_diff2 = self.dm2(mid)
                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
#            return fea1, fea2,fea_diff1,fea_diff2,c1,c2,hdr
            return hdr

class HDRnet_14(nn.Module):
    def __init__(self, args):
        super(HDRnet_14, self).__init__()
        self.args = args
        self.nfeat = self.args.nfeat
        from model.net.CIDNet import CIDNet
        self.net = CIDNet()

    def forward(self, x):
        sample = x[0]
        mid = sample['mid_image']
        gt = sample['hdr_gt']
        mid1 = mid[:,:3,:,:]
        if self.training:
            output_rgb = self.net(mid1)
            gt_rgb = gt
            output_hvi = self.net.HVIT(output_rgb)
            gt_hvi = self.net.HVIT(gt_rgb)
            return output_rgb, output_hvi, gt_hvi 
        else:  
            output = self.net(mid1)
            return output 
#restormer
class HDRnet_15(nn.Module):
    def __init__(self, args):
        super(HDRnet_15, self).__init__()
        self.args = args
        self.nfeat = self.args.nfeat
        self.net = Restormer()

    def forward(self, x):
        sample = x[0]
        mid = sample['mid_image']
        gt = sample['hdr_gt']
        h, w = mid.shape[2], mid.shape[3]
        h = h - h % 16
        w = w - w % 16
        mid = mid[:, :, :h, :w]
        gt = gt[:, :, :h, :w]
        if self.training:
            hdr = self.net(mid)
            return hdr
        else:  
            hdr = self.net(mid)
            return hdr

class HDRnet_16(nn.Module):
    def __init__(self, args):
        super(HDRnet_16, self).__init__()
        self.args = args
        timesteps = 4
        self.nfeat = self.args.nfeat
        self.unet_l = Unet_main_time(self.nfeat)
        self.unet_h = Unet_main_time(self.nfeat)
        self.encoder_l = Encoder_gt(feats=64)
        self.encoder_h = Encoder_gt(feats=64)
        self.condition_l = Encoder_lr(feats=64)
        self.condition_h = Encoder_lr(feats=64)
        self.head_l=Enhance4(256)
        self.head_h=Enhance4(256)
        self.merge = Merginglayer()
        self.weights_initialized = True
        self.should_reset_conditions =False 

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, x):
        sample = x[0]
        diffusion = x[1]
        low = sample['low_image']
        mid = sample['mid_image']
        high = sample['high_image']
        gt = sample['hdr_gt']
        h, w = mid.shape[2], mid.shape[3]
        h = h - h % 16
        w = w - w % 16
        low = low[:, :, :h, :w]
        mid = mid[:, :, :h, :w]
        high = high[:, :, :h, :w]
        gt = gt[:, :, :h, :w]
        expos = sample['expos']
        expos_att = sample['expos_att']
        expos_att = expos_att.float()
        if self.training:
            if not diffusion:
                ## feature map
                fea1 = self.encoder_l(mid, low)# xiang liang
                fea2 = self.encoder_h(mid, high)
                fea_diff1 = fea1
                fea_diff2 = fea2
                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
                l_f2 = None
                h_f2 = None
            else:
                self.freeze_module(self.encoder_l)
                self.freeze_module(self.encoder_h)              
                # GT +mid
                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)
                fea1 = self.head_l(fea1,expos_att,True)
                fea2 = self.head_h(fea2,expos_att,False)
                #mid
                fea_diff1 = self.condition_l(mid)
                fea_diff2 = self.condition_h(mid)
                l_f2 = self.head_l(fea_diff1,expos_att,True)
                h_f2 = self.head_h(fea_diff2,expos_att,False)

                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)
               
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)

            return fea_diff1, fea_diff2, fea1, fea2, _low, _high, hdr,l_f2,h_f2
        else:  
            if not diffusion:
                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)
                fea_diff1 = fea1
                fea_diff2 = fea2

                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)
        
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
            else:
                fea1 = self.encoder_l(mid, low) 
                fea2 = self.encoder_h(mid, high)

                fea_diff1 = self.condition_l(mid)
                fea_diff2 = self.condition_h(mid)
                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)

                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
            #return fea_diff1, fea_diff2, fea1, fea2,hdr
            return _low, _high, hdr


class HDRnet_swinIR(nn.Module):
    def __init__(self, args):
        super(HDRnet_swinIR, self).__init__()
        self.args = args
        os.environ["OMP_NUM_THREADS"] = "12"
        os.environ["MKL_NUM_THREADS"] = "12"
        torch.set_num_threads(12)
        timesteps = 4
        self.nfeat = self.args.nfeat
        window_size = 8
        upscale = 1
        height = (128 // upscale // window_size + 1) * window_size
        width = (128 // upscale // window_size + 1) * window_size
        self.unet_l = SwinIR(upscale=1, img_size=(height, width),
                   window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
        self.unet_h = SwinIR(upscale=1, img_size=(height, width),
                   window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
        self.encoder_l = Encoder_gt(feats=64)
        self.encoder_h = Encoder_gt(feats=64)
        self.condition = Encoder_lr(feats=64)
        self.denoise = denoise(feats=64, timesteps=4)
        self.dm1 = diffusion.DDPM(denoise=self.denoise,
                                   condition=self.condition, feats=64, timesteps=timesteps)
        self.dm2 = diffusion.DDPM(denoise=self.denoise,
                                   condition=self.condition, feats=64, timesteps=timesteps)
        self.merge = Merginglayer()

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, x):
        sample = x[0]
        diffusion = x[1]
        low = sample['low_image']
        mid = sample['mid_image']
        high = sample['high_image']
        gt = sample['hdr_gt']
        expos = sample['expos']
        expos_att = sample['expos_att']

        if self.training:
            if not diffusion:
                ## feature map
                fea1 = self.encoder_l(mid, low)# xiang liang
                fea2 = self.encoder_h(mid, high)
                fea_diff1 = fea1
                fea_diff2 = fea2
                # mid_fea1 = self.condition1(mid)
                # mid_fea2 = self.condition2(mid)
                #_low = self.unet_l(mid, fea_diff1,expos_att,True)
               # _high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)
                #_low,m,m2,m3,m4,m5,R1_out,R2_out,R3_out,R4_out,R5_out = self.unet_l(mid, fea_diff1)
                #_high,n,n2,n3,n4,n5,Rn1_out,Rn2_out,Rn3_out,Rn4_out,Rn5_out = self.unet_h(mid, fea_diff2)
                # _low = self.decoder1(_low)
                # _high = self.decoder2(_high)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
            else:
                self.freeze_module(self.encoder_l)
                self.freeze_module(self.encoder_h)

                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)

                fea_diff1 = self.dm1(mid, fea1)
                fea_diff2 = self.dm2(mid, fea2)
                #_low = self.unet_l(mid, fea_diff1,expos_att,True)
                #_high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)
                #_low ,m,m2,m3,m4,m5,R1_out,R2_out,R3_out,R4_out,R5_out= self.unet_l(mid, fea_diff1)
                #_high,n,n2,n3,n4,n5,Rn1_out,Rn2_out,Rn3_out,Rn4_out,Rn5_out = self.unet_h(mid, fea_diff2)

                #
                # _low = self.decoder1(_low)
                # _high = self.decoder2(_high)
                #
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)

            return fea_diff1, fea_diff2, fea1, fea2, _low, _high, hdr
        else:
             
            if not diffusion:
                fea1 = self.encoder_l(mid, low)  # b,1,h,w
                fea2 = self.encoder_h(mid, high)
                fea_diff1 = fea1
                fea_diff2 = fea2

                #_low = self.unet_l(mid, fea_diff1,expos_att,True)
                #_high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)
                #_low,m,m2,m3,m4,m5,R1_out,R2_out,R3_out,R4_out,R5_out = self.unet_l(mid, fea_diff1)
                #_high,n,n2,n3,n4,n5,Rn1_out,Rn2_out,Rn3_out,Rn4_out,Rn5_out = self.unet_h(mid, fea_diff2)
        
                # _low = self.decoder1(_low)
                # _high = self.decoder2(_high)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
            else:
                # mid_fea1 = self.condition1(mid)
                # mid_fea2 = self.condition2(mid)
                fea_diff1 = self.dm1(mid)
                fea_diff2 = self.dm2(mid)
                #_low = self.unet_l(mid, fea_diff1,expos_att,True)
                #_high = self.unet_h(mid, fea_diff2,expos_att,False)
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)
                #_low ,m,m2,m3,m4,m5,R1_out,R2_out,R3_out,R4_out,R5_out= self.unet_l(mid, fea_diff1)
                #_high,n,n2,n3,n4,n5,Rn1_out,Rn2_out,Rn3_out,Rn4_out,Rn5_out= self.unet_h(mid, fea_diff2)
                # _low = self.decoder1(_low)
                # _high = self.decoder2(_high)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
            return hdr
            
class HDRnet_CD(nn.Module):
    def __init__(self, args):
        super(HDRnet_CD, self).__init__()
        self.args = args
#        os.environ["OMP_NUM_THREADS"] = "12"
#        os.environ["MKL_NUM_THREADS"] = "12"
#        torch.set_num_threads(12)
        timesteps = 4
        self.nfeat = self.args.nfeat
        self.unet_l = CDFormer_SR(upscale=1)
        self.unet_h = CDFormer_SR(upscale=1)
        self.encoder_l = Encoder_gt(feats=64)
        self.encoder_h = Encoder_gt(feats=64)
        self.condition = Encoder_lr(feats=64)
        self.denoise = denoise(feats=64, timesteps=4)
        self.dm1 = diffusion.DDPM(denoise=self.denoise,
                                   condition=self.condition, feats=64, timesteps=timesteps)
        self.dm2 = diffusion.DDPM(denoise=self.denoise,
                                   condition=self.condition, feats=64, timesteps=timesteps)
        self.merge = Merginglayer()

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, x):
        sample = x[0]
        diffusion = x[1]
        low = sample['low_image']
        mid = sample['mid_image']
        high = sample['high_image']
        gt = sample['hdr_gt']
        expos = sample['expos']
        expos_att = sample['expos_att']

        if self.training:
            if not diffusion:
                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)
                fea_diff1 = fea1
                fea_diff2 = fea2

                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)

                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
            else:
                self.freeze_module(self.encoder_l)
                self.freeze_module(self.encoder_h)

                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)

                fea_diff1 = self.dm1(mid, fea1)
                fea_diff2 = self.dm2(mid, fea2)
                
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)

            return fea_diff1, fea_diff2, fea1, fea2, _low, _high, hdr
        else:
             
            if not diffusion:
                fea1 = self.encoder_l(mid, low)  # b,1,h,w
                fea2 = self.encoder_h(mid, high)
                fea_diff1 = fea1
                fea_diff2 = fea2
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
            else:
                fea_diff1 = self.dm1(mid)
                fea_diff2 = self.dm2(mid)
                _low = self.unet_l(mid, fea_diff1)
                _high = self.unet_h(mid, fea_diff2)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
            return hdr


#### no 2 jieduan zhijie shiyong mid
class HDRnet_17(nn.Module):
    def __init__(self, args):
        super(HDRnet_17, self).__init__()
        self.args = args
        timesteps = 4
        self.nfeat = self.args.nfeat
        self.unet_l = Unet_main_time(self.nfeat)
        self.unet_h = Unet_main_time(self.nfeat)
        self.encoder_l = Encoder_gt(feats=64)
        self.encoder_h = Encoder_gt(feats=64)
        self.condition_l = Encoder_lr(feats=64)
        self.condition_h = Encoder_lr(feats=64)
        self.head_l=Enhance4(256)
        self.head_h=Enhance4(256)
        self.merge = Merginglayer()

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, x):
        sample = x[0]
        diffusion = x[1]
        low = sample['low_image']
        mid = sample['mid_image']
        high = sample['high_image']
        gt = sample['hdr_gt']
        h, w = mid.shape[2], mid.shape[3]
        h = h - h % 16
        w = w - w % 16
        low = low[:, :, :h, :w]
        mid = mid[:, :, :h, :w]
        high = high[:, :, :h, :w]
        gt = gt[:, :, :h, :w]
        expos = sample['expos']
        expos_att = sample['expos_att']
        expos_att = expos_att.float()
        if self.training:
            if not diffusion:
                ## feature map
                fea1 = self.encoder_l(mid, low)# xiang liang
                fea2 = self.encoder_h(mid, high)
                fea_diff1 = fea1
                fea_diff2 = fea2
                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
                l_f2 = None
                h_f2 = None
            else:           
                #mid
                fea_diff1 = self.condition_l(mid)
                fea_diff2 = self.condition_h(mid)
                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)

            return _low, _high, hdr
        else:  
            if not diffusion:
                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)
                fea_diff1 = fea1
                fea_diff2 = fea2
                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
            else:
                fea_diff1 = self.condition_l(mid)
                fea_diff2 = self.condition_h(mid)
                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
            #return fea_diff1, fea_diff2, fea1, fea2,hdr
            return hdr,_low,_high  


### transformer_notime        
class HDRnet_18(nn.Module):
    def __init__(self, args):
        super(HDRnet_18, self).__init__()
        self.args = args
        timesteps = 4
        self.nfeat = self.args.nfeat
        self.unet_l = Unet_main_time(self.nfeat)
        self.unet_h = Unet_main_time(self.nfeat)
        self.encoder_l = Encoder_gt(feats=64)
        self.encoder_h = Encoder_gt(feats=64)
        self.condition_l = Encoder_lr(feats=64)
        self.condition_h = Encoder_lr(feats=64)
        self.head_l=Enhance1(256)
        self.head_h=Enhance1(256)
        self.merge = Merginglayer()

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, x):
        sample = x[0]
        diffusion = x[1]
        low = sample['low_image']
        mid = sample['mid_image']
        high = sample['high_image']
        gt = sample['hdr_gt']
        expos = sample['expos']
        expos_att = sample['expos_att']
        expos_att = expos_att.float()
        if self.training:
            if not diffusion:
                ## feature map
                fea1 = self.encoder_l(mid, low)# xiang liang
                fea2 = self.encoder_h(mid, high)
                fea_diff1 = fea1
                fea_diff2 = fea2
                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
                l_f2 = None
                h_f2 = None
            else:
                self.freeze_module(self.encoder_l)
                self.freeze_module(self.encoder_h)              
                # GT +mid
                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)
                
                fea1 = self.head_l(fea1,expos_att,True)
                fea2 = self.head_h(fea2,expos_att,False)
                #mid
                fea_diff1 = self.condition_l(mid)
                fea_diff2 = self.condition_h(mid)
                l_f2 = self.head_l(fea_diff1,expos_att,True)
                h_f2 = self.head_h(fea_diff2,expos_att,False)

                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)

                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)

            return fea_diff1, fea_diff2, fea1, fea2, _low, _high, hdr,l_f2,h_f2
        else:  
            if not diffusion:
                fea1 = self.encoder_l(mid, low)
                fea2 = self.encoder_h(mid, high)

                fea_diff1 = fea1
                fea_diff2 = fea2

                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)
        
                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
            else:
                fea1 = self.encoder_l(mid, low) 
                fea2 = self.encoder_h(mid, high)

                fea_diff1 = self.condition_l(mid)
                fea_diff2 = self.condition_h(mid)
                _low = self.unet_l(mid, fea_diff1,expos_att,True)
                _high = self.unet_h(mid, fea_diff2,expos_att,False)

                hdr = self.merge(_low, mid, _high, expos, gamma=2.2)
            return hdr  