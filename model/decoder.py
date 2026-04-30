from torch import nn
import torch


class Channelupsample(nn.Module):
    def __init__(self, channel):
        super(Channelupsample, self).__init__()
        self.conv1 = nn.Conv2d(3, channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channel, channel*2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(channel*2, channel*4, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = torch.sigmoid(
            self.conv3(
                self.relu(
                    self.conv2(
                        self.relu(self.conv1(x))
                    )
                )
            )
        )

        return x


# pixel shuffle
class upSample(nn.Module):
    def __init__(self, in_channel, out_channel, upscale_factor=2):
        super(upSample, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel * (upscale_factor ** 2), kernel_size=3, padding=1)
        self.deconv = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x):
        conv = self.conv(x)
        deconv_output = self.deconv(conv)
        return deconv_output


class Conv_res(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv_res, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.conv_block = nn.Sequential(*layers)
        self.conv_res = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        m = self.conv_block(x)
        y = self.conv_res(x)
        return m + y


class decoder(nn.Module):
    def __init__(self, ch=64):
        super(decoder, self).__init__()
        # self.feature_transform = feature_transform(ch)
        # add channel do not change size
        self.ChannelUpsample = Channelupsample(ch)
        self.block1 = Conv_res(ch*4, ch*4)
        #self.block11 = Conv_res(ch*4, ch*4)
        self.up1 = upSample(ch*4, ch*2)
        self.block2 = Conv_res(ch*2, ch*2)
        #elf.block22 = Conv_res(ch*2, ch*2)
        self.up2 = upSample(ch*2, ch)
        self.block3 = Conv_res(ch, ch)
        #self.block33 = Conv_res(ch, ch)
        self.up3 = upSample(ch, ch)
        self.conv1 = nn.Conv2d(ch, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, pre):
        # feature1, feature2, feature3 = self.feature_transform(x)
        y = self.ChannelUpsample(pre)
        y = self.block1(y)
        y = self.up1(y)
        y = self.block2(y)
        y = self.up2(y)
        y = self.block3(y)
        y = self.up3(y)
        y = torch.sigmoid(self.conv1(y))

        return y
