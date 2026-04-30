import torch
from torch import nn
from torch.nn import functional as F
import math
from einops import rearrange


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y*x

# pixel shuffle
class UpSample1(nn.Module):
    def __init__(self, in_channel, out_channel, upscale_factor=2):
        super(UpSample1, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.deconv = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x1, x2):
        conv = self.conv(x1)
        deconv_output = self.deconv(conv)
        return torch.cat([deconv_output, x2], dim=1)


class Conv_DecoderBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Conv_DecoderBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            #nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ConvBlock_diff(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, temb_channels=256):
        super(ConvBlock_diff, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        self.conv_block = nn.Sequential(*layers)

        # 如果需要处理时间嵌入，则定义一个时间嵌入投影层
        self.temb_channels = temb_channels
        self.temb_proj = nn.Sequential(nn.Linear(temb_channels, out_channels),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(out_channels, out_channels))

    def forward(self, x, temb):
        h = x
        # 处理第一个卷积层

        h = self.conv_block[0](h)  # 第一个卷积层的卷积操作
        temb_proj = self.temb_proj(temb).view(-1, h.size(1), 1, 1)  # 映射并调整形状
        h = h + temb_proj  # 将时间步嵌入与卷积输出相加
        h = self.conv_block[1](h)  # 第一个 ReLU 激活

        # 遍历后续的卷积层
        for i in range(2, len(self.conv_block), 2):
            h = self.conv_block[i](h)  # 卷积
            h = self.conv_block[i + 1](h)  # 激活
        return h


class MaxPoolLayer(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(MaxPoolLayer, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        return self.pool(x)


def nonlinearity(x):
    return x*torch.sigmoid(x)


def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # Zero pad
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb


class TimestepEmbedding(nn.Module):
    def __init__(self, nfeat):
        super(TimestepEmbedding, self).__init__()
        self.dense = nn.ModuleList([
            nn.Linear(nfeat, nfeat * 4),
            nn.Linear(nfeat * 4, nfeat * 4),
        ])

    def forward(self, t):
        t = self.dense[0](t)
        # 非线性激活函数
        t = nonlinearity(t)
        t = self.dense[1](t)
        return t


class Unet_diffusion(nn.Module):
    def __init__(self, nfeat=64):
        super(Unet_diffusion, self).__init__()
        self.nfeat = nfeat

        self.encoder1 = ConvBlock_diff(in_channels=2, out_channels=64, num_layers=2)
        self.pool1 = MaxPoolLayer()
        self.encoder2 = ConvBlock_diff(in_channels=64, out_channels=128, num_layers=2)
        self.pool2 = MaxPoolLayer()
        self.encoder3 = ConvBlock_diff(in_channels=128, out_channels=256, num_layers=3)

        self.pool3 = MaxPoolLayer()
        self.encoder4 = ConvBlock_diff(in_channels=256, out_channels=256, num_layers=3)

        # timestep embedding
        self.temb = TimestepEmbedding(nfeat=self.nfeat)

        self.se1 = SEBlock(self.nfeat)
        self.se2 = SEBlock(self.nfeat * 2)
        self.se3 = SEBlock(self.nfeat * 4)
        self.se4 = SEBlock(self.nfeat*4)

        # 调整通道数3 -> 64
        self.conv1x1_1 = nn.Conv2d(2, 64, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(64, 128, kernel_size=1)
        self.conv1x1_3 = nn.Conv2d(128, 256, kernel_size=1)
        self.conv1x1_4 = nn.Conv2d(256, 256, kernel_size=1)

        self.u1 = UpSample1(256, 256)
        self.c6 = Conv_DecoderBlock(512, 256)

        self.u2 = UpSample1(256, 128)
        self.c7 = Conv_DecoderBlock(256, 128)

        self.u3 = UpSample1(128, 64)
        self.c8 = Conv_DecoderBlock(128, 64)

        self.out = nn.Sequential(nn.Conv2d(64, 3, kernel_size=3, padding=1),
                         nn.Conv2d(3, 1, kernel_size=3, padding=1))

    def forward(self, x, t):
        temb = get_timestep_embedding(t, self.nfeat)
        temb = self.temb(temb)

        R1_in = x
        R1_out = self.encoder1(R1_in, temb)
        R1_se = self.se1(R1_out)  # att
        R1_ad = self.conv1x1_1(R1_in)
        R1_res = R1_ad + R1_se  # res
        R1_out = self.pool1(R1_res)

        R2_in = R1_out
        R2_out = self.encoder2(R2_in, temb)
        R2_se = self.se2(R2_out)
        R2_ad = self.conv1x1_2(R2_in)
        R2_res = R2_ad + R2_se
        R2_out = self.pool2(R2_res)

        R3_in = R2_out
        R3_out = self.encoder3(R3_in, temb)
        R3_se = self.se3(R3_out)
        R3_ad = self.conv1x1_3(R3_in)
        R3_res = R3_ad + R3_se
        R3_out = self.pool3(R3_res)

        R4_in = R3_out
        R4_out = self.encoder4(R4_in, temb)
        R4_se = self.se4(R4_out)
        R4_ad = self.conv1x1_4(R4_in)
        R4_res = R4_ad + R4_se
        R4_out = R4_res

        O1 = self.c6(self.u1(R4_out, R3_res))
        O2 = self.c7(self.u2(O1, R2_res))
        O3 = self.c8(self.u3(O2, R1_res))

        out = torch.sigmoid(self.out(O3))

        return out


class feature_attention(nn.Module):
    def __init__(self, nfeat):
        super(feature_attention, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, nfeat, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(nfeat),
                                  nn.Conv2d(nfeat, nfeat, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(nfeat),
                                  nn.Conv2d(nfeat, nfeat, kernel_size=3, stride=1, padding=1))
        self.bn = nn.BatchNorm2d(nfeat)

    def forward(self, x, fea):
        fea = self.conv(fea)
        x = fea*self.bn(x) + fea
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers):
        super(ConvBlock, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
            #layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels  # 后续层的输入通道等于输出通道
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_block(x)


class Unet_main_map(nn.Module):
    def __init__(self, nfeat=64):
        super(Unet_main_map, self).__init__()
        self.nfeat = nfeat

        self.encoder1 = nn.Sequential(
            ConvBlock(in_channels=1, out_channels=64, num_layers=2)
        )
        self.pool1 = MaxPoolLayer()
        self.encoder2 = nn.Sequential(
            ConvBlock(in_channels=64, out_channels=128, num_layers=2),
        )
        self.pool2 = MaxPoolLayer()
        self.encoder3 = nn.Sequential(
            ConvBlock(in_channels=128, out_channels=256, num_layers=3)
        )
        self.pool3 = MaxPoolLayer()
        self.encoder4 = nn.Sequential(
            ConvBlock(in_channels=256, out_channels=512, num_layers=3)
        )
        self.pool4 = MaxPoolLayer()
        self.encoder5 = nn.Sequential(
            ConvBlock(in_channels=512, out_channels=512, num_layers=3)
        )

        self.se1 = SEBlock(self.nfeat)
        self.se2 = SEBlock(self.nfeat*2)
        self.se3 = SEBlock(self.nfeat*4)
        self.se4 = SEBlock(self.nfeat*8)
        self.se5 = SEBlock(self.nfeat*8)

        self.conv1x1_1 = nn.Conv2d(1, 64, kernel_size=1)  # 调整通道数3 -> 64
        self.conv1x1_2 = nn.Conv2d(64, 128, kernel_size=1)  # 调整通道数64 -> 128,并改变尺寸大小
        self.conv1x1_3 = nn.Conv2d(128, 256, kernel_size=1)
        self.conv1x1_4 = nn.Conv2d(256, 512, kernel_size=1)
        self.conv1x1_5 = nn.Conv2d(512, 512, kernel_size=1)

        self.fea1 = feature_attention(self.nfeat)
        # self.fea2 = feature_attention(self.nfeat*2)
        # self.fea3 = feature_attention(self.nfeat*4)
        # self.fea4 = feature_attention(self.nfeat*8)

        self.u1 = UpSample1(512, 512)
        self.c6 = Conv_DecoderBlock(1024, 512)

        self.u2 = UpSample1(512, 256)
        self.c7 = Conv_DecoderBlock(512, 256)

        self.u3 = UpSample1(256, 128)
        self.c8 = Conv_DecoderBlock(256, 128)

        self.u4 = UpSample1(128, 64)
        self.c9 = Conv_DecoderBlock(128, 64)

        self.out = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x, fea): # shape of x and fea is (8,1,16,16)
        R1_in = x
        R1_out = self.encoder1(R1_in)
        R1_out = self.fea1(R1_out, fea)
        R1_se = self.se1(R1_out)  # att
        R1_ad = self.conv1x1_1(R1_in)
        R1_res = R1_ad + R1_se  # res
        R1_out = self.pool1(R1_res) #(8,64,8,8)
        
        R2_in = R1_out
        R2_out = self.encoder2(R2_in) # (8,128,8,8)
        R2_se = self.se2(R2_out)
        R2_ad = self.conv1x1_2(R2_in)
        R2_res = R2_ad + R2_se
        R2_out = self.pool2(R2_res)

        R3_in = R2_out
        R3_out = self.encoder3(R3_in)
        R3_se = self.se3(R3_out)
        R3_ad = self.conv1x1_3(R3_in)
        R3_res = R3_ad + R3_se
        R3_out = self.pool3(R3_res)

        R4_in = R3_out
        R4_out = self.encoder4(R4_in)
        R4_se = self.se4(R4_out)
        R4_ad = self.conv1x1_4(R4_in)
        R4_res = R4_ad + R4_se
        R4_out = self.pool4(R4_res)

        R5_in = R4_out
        R5_out = self.encoder5(R5_in)
        R5_se = self.se5(R5_out)
        R5_ad = self.conv1x1_5(R5_in)
        R5_res = R5_ad + R5_se
        R5_out = R5_res

        O1 = self.c6(self.u1(R5_out, R4_res))
        O2 = self.c7(self.u2(O1, R3_res))
        O3 = self.c8(self.u3(O2, R2_res))
        O4 = self.c9(self.u4(O3, R1_res))

        out = self.out(O4)

        return out


class expos_se(nn.Module):  #处理曝光相关特征
    def __init__(self, nFeat):
        super(expos_se, self).__init__()
        self.nFeat = nFeat
        self.fc = nn.Sequential(
            nn.Linear(self.nFeat * 5, self.nFeat * 8),
            nn.Linear(self.nFeat * 8, self.nFeat * 4),
            nn.Linear(self.nFeat * 4, self.nFeat),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        # print("x",x.shape)
        # print("y",y.shape)
        a1, b1, c1, d1 = x.size()  # seblock
        a2, b2, c2, d2 = y.size()  # expos_att
        z = torch.cat([x, y], dim=1).view(a1, b1+b2)
        z = self.fc(z)
        return z.view(a1, b1, c1, d1)


class CDIM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.kernel = nn.Sequential(
            nn.Linear(256, dim*2, bias=False),
        )

    def forward(self, x, cdp):
        """
        Input: x: (B, H*W, C), x_size: (H, W)
        Output: x: (B, H*W, C)
        """    
        x = self.norm(x)    
        B, N, C = x.shape
        cdp=self.kernel(cdp).view(-1,1,C*2)
        cdp1,cdp2=cdp.chunk(2, dim=2)
        x = x*cdp1+cdp2  
        return x


class Unet_main(nn.Module):
    def __init__(self, nfeat):
        super(Unet_main, self).__init__()
        self.nfeat = nfeat

        self.encoder1 = nn.Sequential(
            ConvBlock(in_channels=6, out_channels=self.nfeat, num_layers=2)
        )
        self.pool1 = MaxPoolLayer()
        self.encoder2 = nn.Sequential(
            ConvBlock(in_channels=self.nfeat, out_channels=self.nfeat*2, num_layers=2),
        )
        self.pool2 = MaxPoolLayer()
        self.encoder3 = nn.Sequential(
            ConvBlock(in_channels=self.nfeat*2, out_channels=self.nfeat*4, num_layers=3)
        )
        self.pool3 = MaxPoolLayer()
        self.encoder4 = nn.Sequential(
            ConvBlock(in_channels=self.nfeat*4, out_channels=self.nfeat*8, num_layers=3)
        )
        self.pool4 = MaxPoolLayer()
        self.encoder5 = nn.Sequential(
            ConvBlock(in_channels=self.nfeat*8, out_channels=self.nfeat*8, num_layers=3)
        )

        self.se1 = SEBlock(self.nfeat)
        self.se2 = SEBlock(self.nfeat * 2)
        self.se3 = SEBlock(self.nfeat * 4)
        self.se4 = SEBlock(self.nfeat * 8)
        self.se5 = SEBlock(self.nfeat * 8)

        self.cdim = CDIM(self.nfeat)
        self.cdim2 = CDIM(self.nfeat*2)
        self.cdim3 = CDIM(self.nfeat*4)
        self.cdim4  = CDIM(self.nfeat*8)
        self.cdim5 = CDIM(self.nfeat*8)
        
        self.int = expos_se(self.nfeat)
        self.norm = nn.LayerNorm(self.nfeat)
        self.norm2 = nn.LayerNorm(self.nfeat*2)
        self.norm3 = nn.LayerNorm(self.nfeat*4)
        self.norm4 = nn.LayerNorm(self.nfeat*8)
        self.norm5 = nn.LayerNorm(self.nfeat*8)
        self.conv1x1_1 = nn.Conv2d(6, self.nfeat, kernel_size=1)  # 调整通道数3 -> 64
        self.conv1x1_2 = nn.Conv2d(self.nfeat, self.nfeat*2, kernel_size=1)  # 调整通道数64 -> 128,
        self.conv1x1_3 = nn.Conv2d(self.nfeat*2, self.nfeat*4, kernel_size=1)
        self.conv1x1_4 = nn.Conv2d(self.nfeat*4, self.nfeat*8, kernel_size=1)
        self.conv1x1_5 = nn.Conv2d(self.nfeat*8, self.nfeat*8, kernel_size=1)

        #self.fea1 = feature_attention(self.nfeat)
        # self.fea2 = feature_attention(self.nfeat*2)
        # self.fea3 = feature_attention(self.nfeat*4)
        # self.fea4 = feature_attention(self.nfeat*8)

        self.u1 = UpSample1(self.nfeat*8, self.nfeat*8)
        self.c6 = Conv_DecoderBlock(self.nfeat*16, self.nfeat*8)

        self.u2 = UpSample1(self.nfeat*8, self.nfeat*4)
        self.c7 = Conv_DecoderBlock(self.nfeat*8, self.nfeat*4)

        self.u3 = UpSample1(self.nfeat*4, self.nfeat*2)
        self.c8 = Conv_DecoderBlock(self.nfeat*4, self.nfeat*2)

        self.u4 = UpSample1(self.nfeat*2, self.nfeat)
        self.c9 = Conv_DecoderBlock(self.nfeat*2, self.nfeat)

        self.out = nn.Conv2d(self.nfeat, 3, kernel_size=3, padding=1)

    def forward(self, x, fea):  # shape of x and fea is 
        ####fea = fea.unsqueeze(-1).unsqueeze(-1)
        R1_in = x
        m = self.encoder1(R1_in)
        _, _, H, W = m.shape
        x_size = [H, W]
        R1_out = rearrange(m, "b c h w -> b (h w) c")
        R1_out = self.cdim(R1_out,fea)
        #R1_out = self.norm(R1_out)
        R1_out = rearrange(R1_out, "b (h w) c -> b c h w", h=H, w=W)
        R1_se = self.se1(R1_out)  # att  shu chu quan zhong
        ####R1_se = self.int(R1_se, fea)
        ####R1_se = torch.mul(R1_se,R1_out)
            
        R1_ad = self.conv1x1_1(R1_in)
        R1_res = R1_ad + R1_se  # res

        R1_ = self.pool1(R1_res)  # 
        R2_in = R1_
        m2 = self.encoder2(R2_in)  # 
        _, _, H, W = m2.shape
        x_size = [H, W]
        R2_out = rearrange(m2, "b c h w -> b (h w) c")
        R2_out = self.cdim2(R2_out,fea)
       # R2_out = self.norm2(R2_out)
        R2_out = rearrange(R2_out, "b (h w) c -> b c h w", h=H, w=W)
        R2_se = self.se2(R2_out)
        R2_ad = self.conv1x1_2(R2_in)
        R2_res = R2_ad + R2_se
        R2_ = self.pool2(R2_res)
        
        R3_in = R2_
        m3 = self.encoder3(R3_in)
        _, _, H, W = m3.shape
        x_size = [H, W]
        R3_out = rearrange(m3, "b c h w -> b (h w) c")
        R3_out = self.cdim3(R3_out,fea)
        #R3_out = self.norm3(R3_out)
        R3_out = rearrange(R3_out, "b (h w) c -> b c h w", h=H, w=W)
        R3_se = self.se3(R3_out)
        R3_ad = self.conv1x1_3(R3_in)
        R3_res = R3_ad + R3_se
        R3_= self.pool3(R3_res)

        R4_in = R3_
        m4 = self.encoder4(R4_in)
        _, _, H, W = m4.shape
        x_size = [H, W]
        R4_out = rearrange(m4, "b c h w -> b (h w) c")
        R4_out = self.cdim4(R4_out,fea)
        #R4_out = self.norm4(R4_out)
        R4_out = rearrange(R4_out, "b (h w) c -> b c h w", h=H, w=W)
        R4_se = self.se4(R4_out)
        R4_ad = self.conv1x1_4(R4_in)
        R4_res = R4_ad + R4_se
        R4_ = self.pool4(R4_res)

        R5_in = R4_
        m5 = self.encoder5(R5_in)
        _, _, H, W = m5.shape
        x_size = [H, W]
        R5_out = rearrange(m5, "b c h w -> b (h w) c")
        R5_out = self.cdim5(R5_out,fea)
        #R5_out = self.norm5(R5_out)
        R5_out = rearrange(R5_out, "b (h w) c -> b c h w", h=H, w=W)
        R5_se = self.se5(R5_out)
        R5_ad = self.conv1x1_5(R5_in)
        R5_res = R5_ad + R5_se
        R5_ = R5_res

        O1 = self.c6(self.u1(R5_, R4_res))
        O2 = self.c7(self.u2(O1, R3_res))
        O3 = self.c8(self.u3(O2, R2_res))
        O4 = self.c9(self.u4(O3, R1_res))

        out = torch.sigmoid(self.out(O4))

        return out

class Unet_main_time(nn.Module):
    def __init__(self, nfeat):
        super(Unet_main_time, self).__init__()
        self.nfeat = nfeat
        self.encoder1 = nn.Sequential(
            ConvBlock(in_channels=7, out_channels=self.nfeat, num_layers=2)
        )
        self.pool1 = MaxPoolLayer()
        self.encoder2 = nn.Sequential(
            ConvBlock(in_channels=self.nfeat, out_channels=self.nfeat*2, num_layers=2),
        )
        self.pool2 = MaxPoolLayer()
        self.encoder3 = nn.Sequential(
            ConvBlock(in_channels=self.nfeat*2, out_channels=self.nfeat*4, num_layers=3)
        )
        self.pool3 = MaxPoolLayer()
        self.encoder4 = nn.Sequential(
            ConvBlock(in_channels=self.nfeat*4, out_channels=self.nfeat*8, num_layers=3)
        )
        self.pool4 = MaxPoolLayer()
        self.encoder5 = nn.Sequential(
            ConvBlock(in_channels=self.nfeat*8, out_channels=self.nfeat*8, num_layers=3)
        )

        self.se1 = SEBlock(self.nfeat)
        self.se2 = SEBlock(self.nfeat * 2)
        self.se3 = SEBlock(self.nfeat * 4)
        self.se4 = SEBlock(self.nfeat * 8)
        self.se5 = SEBlock(self.nfeat * 8)

        self.cdim = CDIM(self.nfeat)
        self.cdim2 = CDIM(self.nfeat*2)
        self.cdim3 = CDIM(self.nfeat*4)
        self.cdim4  = CDIM(self.nfeat*8)
        self.cdim5 = CDIM(self.nfeat*8)
        
        #self.int = expos_se(self.nfeat)
        self.norm = nn.LayerNorm(self.nfeat)
        self.norm2 = nn.LayerNorm(self.nfeat*2)
        self.norm3 = nn.LayerNorm(self.nfeat*4)
        self.norm4 = nn.LayerNorm(self.nfeat*8)
        self.norm5 = nn.LayerNorm(self.nfeat*8)
        self.conv1x1_1 = nn.Conv2d(7, self.nfeat, kernel_size=1)  # 调整通道数3 -> 64
        self.conv1x1_2 = nn.Conv2d(self.nfeat, self.nfeat*2, kernel_size=1)  # 调整通道数64 -> 128,并改变尺寸大小
        self.conv1x1_3 = nn.Conv2d(self.nfeat*2, self.nfeat*4, kernel_size=1)
        self.conv1x1_4 = nn.Conv2d(self.nfeat*4, self.nfeat*8, kernel_size=1)
        self.conv1x1_5 = nn.Conv2d(self.nfeat*8, self.nfeat*8, kernel_size=1)

        #self.fea1 = feature_attention(self.nfeat)
        # self.fea2 = feature_attention(self.nfeat*2)
        # self.fea3 = feature_attention(self.nfeat*4)
        # self.fea4 = feature_attention(self.nfeat*8)

        self.u1 = UpSample1(self.nfeat*8,self.nfeat*8)
        self.c6 = Conv_DecoderBlock(self.nfeat*16, self.nfeat*8)

        self.u2 = UpSample1(self.nfeat*8,self.nfeat*4)
        self.c7 = Conv_DecoderBlock(self.nfeat*8, self.nfeat*4)

        self.u3 = UpSample1(self.nfeat*4, self.nfeat*2)
        self.c8 = Conv_DecoderBlock(self.nfeat*4, self.nfeat*2)

        self.u4 = UpSample1(self.nfeat*2, self.nfeat)
        self.c9 = Conv_DecoderBlock(self.nfeat*2, self.nfeat)

        self.out = nn.Conv2d(self.nfeat, 4, kernel_size=3, padding=1)

    def forward(self, x, fea,expos,time):  # shape of x and fea is 
        ####fea = fea.unsqueeze(-1).unsqueeze(-1)
        #add time
        index = 0 if time else 2
        expos_feature = expos[:, index].unsqueeze(1).unsqueeze(2).unsqueeze(3)
        expos_feature = expos_feature.expand(-1, 1, x.shape[2], x.shape[3])
        x = torch.cat((x, expos_feature), dim=1)

        R1_in = x
        m = self.encoder1(R1_in)
        _, _, H, W = m.shape
        x_size = [H, W]
        R1_out = rearrange(m, "b c h w -> b (h w) c")
        R1_out = self.cdim(R1_out,fea)
        R1_out = self.norm(R1_out)
        R1_out = rearrange(R1_out, "b (h w) c -> b c h w", h=H, w=W)

        R1_se = self.se1(R1_out)  # att  shu chu quan zhong
        ####R1_se = self.int(R1_se, fea)
        ####R1_se = torch.mul(R1_se,R1_out)

        R1_ad = self.conv1x1_1(R1_in)
        R1_res = R1_ad + R1_se  # res
        R1_out = self.pool1(R1_res)  # 
        R2_in = R1_out
        m2 = self.encoder2(R2_in)  # 
        _, _, H, W = m2.shape
        x_size = [H, W]
        R2_out = rearrange(m2, "b c h w -> b (h w) c")
        R2_out = self.cdim2(R2_out,fea)
        R2_out = self.norm2(R2_out)
        R2_out = rearrange(R2_out, "b (h w) c -> b c h w", h=H, w=W)
        R2_se = self.se2(R2_out)
        R2_ad = self.conv1x1_2(R2_in)
        R2_res = R2_ad + R2_se
        R2_out = self.pool2(R2_res)
        
        R3_in = R2_out
        m3 = self.encoder3(R3_in)
        _, _, H, W = m3.shape
        x_size = [H, W]
        R3_out = rearrange(m3, "b c h w -> b (h w) c")
        R3_out = self.cdim3(R3_out,fea)
        R3_out = self.norm3(R3_out)
        R3_out = rearrange(R3_out, "b (h w) c -> b c h w", h=H, w=W)
        R3_se = self.se3(R3_out)
        R3_ad = self.conv1x1_3(R3_in)
        R3_res = R3_ad + R3_se
        R3_out = self.pool3(R3_res)

        R4_in = R3_out
        m4 = self.encoder4(R4_in)
        _, _, H, W = m4.shape
        x_size = [H, W]
        R4_out = rearrange(m4, "b c h w -> b (h w) c")
        R4_out = self.cdim4(R4_out,fea)
        R4_out = self.norm4(R4_out)
        R4_out = rearrange(R4_out, "b (h w) c -> b c h w", h=H, w=W)
        R4_se = self.se4(R4_out)
        R4_ad = self.conv1x1_4(R4_in)
        R4_res = R4_ad + R4_se
        R4_out = self.pool4(R4_res)

        R5_in = R4_out
        m5 = self.encoder5(R5_in)
        _, _, H, W = m5.shape
        x_size = [H, W]
        R5_out = rearrange(m5, "b c h w -> b (h w) c")
        R5_out = self.cdim5(R5_out,fea)
        R5_out = self.norm5(R5_out)
        R5_out = rearrange(R5_out, "b (h w) c -> b c h w", h=H, w=W)
        R5_se = self.se5(R5_out)
        R5_ad = self.conv1x1_5(R5_in)
        R5_res = R5_ad + R5_se
        R5_out = R5_res

        O1 = self.c6(self.u1(R5_out, R4_res))
        O2 = self.c7(self.u2(O1, R3_res))
        O3 = self.c8(self.u3(O2, R2_res))
        O4 = self.c9(self.u4(O3, R1_res))
        
        out = self.out(O4)
        img_ch = out[:, :3, :, :]
        expos_ch = out[:, 3:, :, :]
        out = torch.sigmoid(img_ch * expos_ch)

        #out = torch.sigmoid(self.out(O4))

        return out
