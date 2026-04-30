from torch import nn
from util.util import *
from model import unet


class Diffusion(nn.Module):
    def __init__(self, args):
        super(Diffusion, self).__init__()
        self.args = args

        self.device = torch.device('cpu' if args.cpu else 'cuda')
        # 直接传入 Unet 网络，初始化 Unet
        self.Unet = unet.Unet_diffusion(self.args.nfeat)

        # 获取beta调度
        betas = get_beta_schedule(
            beta_schedule=self.args.beta_schedule,
            beta_start=self.args.beta_start,
            beta_end=self.args.beta_end,
            num_diffusion_timesteps=self.args.num_diffusion_timesteps,
            device=self.device
        )
        self.betas = betas.float().to(self.device)
        self.num_timesteps = self.betas.shape[0]

    @staticmethod
    def compute_alpha(beta, t):
        """计算 alpha_t：扩散模型中每个时间步的信号比例"""
        beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
        a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1, 1)
        return a

    def sample_training(self, x_cond, b, eta=0.):
        """训练时的采样：使用逆过程从噪声生成特征"""
        skip = 25  # 假设每次采样时使用 1/10 步
        seq = range(0, 500, skip)
        n, c, h, w = x_cond.shape
        seq_next = [-1] + list(seq[:-1])
        x = torch.randn(n, c, h, w, device=self.device)
        xs = [x]
        for i, j in zip(reversed(seq), reversed(seq_next)):
            t = (torch.ones(n) * i).to(x.device)
            next_t = (torch.ones(n) * j).to(x.device)
            at = self.compute_alpha(b, t.long())

            at_next = self.compute_alpha(b, next_t.long())
            xt = xs[-1].to(x.device)
            et = self.Unet(torch.cat([x_cond, xt], dim=1), t)
            x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()

            c1 = eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
            c2 = ((1 - at_next) - c1 ** 2).sqrt()
            xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(x) + c2 * et
            xs.append(xt_next.to(x.device))
        return xs[-1]

    def forward(self, fea, inputs=None): # 条件 , 目标图像
        """训练和推理阶段的前向传播"""
        data_dict = {}
        # encoder
        fea = data_transform(fea)
        b = self.betas.to(fea.device)

        if self.training:
            inputs = data_transform(inputs)
            # 随机选择时间步 t 来训练 Unet
            t = torch.randint(low=0, high=self.num_timesteps, size=(inputs.shape[0] // 2 + 1,)).to(inputs.device)
            t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:inputs.shape[0]].to(inputs.device)
            a = (1 - b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
            e = torch.randn_like(inputs)
            #  在目标图像加噪声
            x = inputs * a.sqrt() + e * (1.0 - a).sqrt()

            # 使用 Unet 预测噪声  训练过程中 是 条件和 噪声cat
            noise_output = self.Unet(torch.cat([fea, x], dim=1), t.float())

            pre_fea = self.sample_training(fea, b)
            pre_fea = inverse_data_transform(pre_fea)
            data_dict["noise_output"] = noise_output
            data_dict["e"] = e
            data_dict["pre_fea"] = pre_fea

        else:
            # 推理阶段
            pre_fea = self.sample_training(fea, b)
            pre_fea = inverse_data_transform(pre_fea)
            data_dict["pre_fea"] = pre_fea

        return data_dict