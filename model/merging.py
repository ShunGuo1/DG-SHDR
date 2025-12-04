import torch
from torch import nn


class Merginglayer(nn.Module):
    def __init__(self):
        super(Merginglayer, self).__init__()

    @staticmethod
    def lambda_1(x):
        # torch.ones_like(x) 避免内存泄漏
        return torch.where(x < 0.5, torch.ones_like(x), -2 * x + 2)

    @staticmethod
    def lambda_2(x):
        return torch.where(x < 0.5, 2 * x, -2 * x + 2)

    @staticmethod
    def lambda_3(x):
        return torch.where(x < 0.5, 2 * x, torch.ones_like(x))

    @staticmethod
    def exposure_adjust(I, gamma, t):
        return (I ** gamma) / (t + 1e-6)

    def forward(self, I1, I2, I3, expos, gamma):
        hdr_image = torch.zeros_like(I1)
        for c in range(3):  # 每个通道单独处理
            expos1 = expos[:, 0].view(-1, 1, 1)
            expos2 = expos[:, 1].view(-1, 1, 1)
            expos3 = expos[:, 2].view(-1, 1, 1)

            H1 = self.exposure_adjust(I1[:, c, :, :], gamma, expos1)
            H2 = self.exposure_adjust(I2[:, c, :, :], gamma, expos2)
            H3 = self.exposure_adjust(I3[:, c, :, :], gamma, expos3)

            I2_normalized = I2[:, c, :, :]
            alpha_1 = 1 - self.lambda_1(I2_normalized)
            alpha_2 = self.lambda_2(I2_normalized)
            alpha_3 = 1 - self.lambda_3(I2_normalized)

            numerator = alpha_1 * H1 + alpha_2 * H2 + alpha_3 * H3
            denominator = alpha_1 + alpha_2 + alpha_3
            hdr_image[:, c, :, :] = numerator / (denominator + 1e-6)

        return hdr_image

