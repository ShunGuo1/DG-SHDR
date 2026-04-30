from torch import nn
import torch
import torch.nn as nn
import model.common as common


class Encoder_lr(nn.Module):
    def __init__(self, feats=64):
        super(Encoder_lr, self).__init__()
        self.E = nn.Sequential(
            nn.Conv2d(6, feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            nn.Conv2d(feats, feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feats * 2, feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feats * 2, feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(feats * 4, feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats * 4, feats * 4),
            nn.LeakyReLU(0.1, True)
        )
    def reset_parameters(self):
        for module in self.modules():
          
            if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.AdaptiveAvgPool2d)):
                continue
           
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                # print(f"Reset Conv2d layer: {module}")

            # 
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                # print(f"Reset Linear layer: {module}")

            elif isinstance(module, common.ResBlock) and hasattr(module, 'reset_parameters'):
                module.reset_parameters()
                # print(f"Reset ResBlock: {module}")
    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        fea1 = self.mlp(fea)
        return fea1


class Encoder_gt(nn.Module):
    def __init__(self, feats=64):
        super(Encoder_gt, self).__init__()

        self.D = nn.Sequential(
            nn.Conv2d(12, feats, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            nn.Conv2d(feats, feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feats * 2, feats * 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feats * 2, feats * 4, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1)
        )


        self.C = nn.Sequential(
            nn.Conv2d(6, feats, kernel_size=7, stride=7, padding=0),
            nn.LeakyReLU(0.1, True),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            common.ResBlock(common.default_conv, feats, kernel_size=3),
            nn.Conv2d(feats, feats*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feats*2),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(feats*2, feats*4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(feats*4),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(feats * 4 * 2, feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats * 4, feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats * 4, feats * 4),
            nn.LeakyReLU(0.1, True),
            nn.Linear(feats * 4, feats * 4),
            nn.LeakyReLU(0.1, True)
        )

    def forward(self, x, gt):
        x = torch.cat([x, gt], dim=1)
        x1_ave = self.D(x).squeeze(-1).squeeze(-1)
        x2_ave = self.C(gt).squeeze(-1).squeeze(-1)
        fea = self.mlp(torch.cat([x1_ave, x2_ave], dim=1))
        #fea = self.mlp(x1_ave)
        return fea


class MLP(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=512, out_dim=256):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),


            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.net(x)

class MLP_time(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=512, out_dim=128):
        super(MLP_time, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, out_dim)
        )
        self.se_mask = nn.Sequential(
            nn.Linear(1, in_dim),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, x, expos, time):
        index = 0 if time else 2
        expos_mask = self.se_mask(expos[:, index].unsqueeze(1))
        x = x * expos_mask
        return self.net(x)

class MLP_time2(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=768, out_dim=128):
        super(MLP_time2, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim*2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Linear(hidden_dim, out_dim)
        )
        self.se_mask = nn.Sequential(
            nn.Linear(1, in_dim),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

    def forward(self, x, expos, time):
        index = 0 if time else 2
        expos_mask = self.se_mask(expos[:, index].unsqueeze(1))
        return self.net(torch.cat([x,expos_mask], dim=1))

if __name__ == '__main__':
    model = Encoder_lr()
    output = model(torch.randn(1, 3, 32, 32))
    print(output.shape)