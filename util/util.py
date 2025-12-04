import numpy as np
import torch
import os
import torch.optim as optim
import torch.optim.lr_scheduler as lrs
from skimage.metrics import peak_signal_noise_ratio
from datetime import datetime
from option import args
from skimage.metrics import structural_similarity as ssim
import random


def pre_process_png(img):
    img = img.astype(np.float32)
    img = img / 255
    return img


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


def pre_process_tif(img):
    img = img.astype(np.float32)
    img = img / 65535
    return img

def LDR2HDR(LDR, expo):
    return (LDR ** 2.2) / expo
    

def formulate_hdr(x):
    assert len(x.shape) == 4
    _hdr = torch.clamp(x, 0, 1)
    _hdr = torch.round(_hdr[0] * 65535)
    return _hdr


def get_beta_schedule(beta_schedule, beta_start, beta_end, num_diffusion_timesteps, device):#shuai jian han shu
    if beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    else:
        raise NotImplementedError(f"Beta schedule {beta_schedule} not implemented.")
    return torch.tensor(betas).float().to(device)


def data_augmentation(x, method):
    if method == 0:
        return np.rot90(x)
    if method == 1:
        return np.fliplr(x)
    if method == 2:
        return np.flipud(x)
    if method == 3:
        return np.rot90(np.rot90(x))
    if method == 4:
        return np.rot90(np.fliplr(x))
    if method == 5:
        return np.rot90(np.flipud(x))


def list_filter(file_list, tail):
    r = []
    for f in file_list:# 分割文件名和扩展名，并存储在变量 s 中
        s = os.path.splitext(f)# 如果文件的扩展名与指定的尾部字符串匹配，则将文件加入结果列表 r
        if s[1] == tail:
            r.append(f)
    return r #将后缀为tail的文件名放入r并返回


def make_optimizer(args, my_model):

    """lambda x: x.requires_grad是个函数， my_model.parameters()是函数输入的参数
    requires_grad为false时候是为了固定网络的底层，这样在反向过程中就不会计算这些参数对应的梯度
    """

    trainable = filter(lambda x: x.requires_grad, my_model.parameters())

    if args.optimizer == 'SGD':
        optimizer_function = optim.SGD
        kwargs = {'momentum': args.momentum}
    elif args.optimizer == 'ADAM':
        optimizer_function = optim.Adam
        kwargs = {
            'betas': (args.beta1, args.beta2),
            'eps': args.epsilon
        }
    elif args.optimizer == 'RMSprop':
        optimizer_function = optim.RMSprop
        kwargs = {'eps': args.epsilon}

    kwargs['weight_decay'] = args.weight_decay
    kwargs['lr'] = args.lr
    return optimizer_function(trainable, **kwargs)


def make_scheduler(args, my_optimizer):
    """学习速率衰减类型"""
    if args.decay_type == 'step':
        scheduler = lrs.StepLR(
            my_optimizer,
            step_size=args.lr_decay_sr,
            gamma=args.gamma_sr,
        )
    elif args.decay_type.find('step') >= 0:
        milestones = args.decay_type.split('_')
        milestones.pop(0)
        milestones = list(map(lambda x: int(x), milestones))
        scheduler = lrs.MultiStepLR(
            my_optimizer,
            milestones=milestones,
            gamma=args.gamma
        )

    elif args.decay_type == 'cosine':
        scheduler = lrs.CosineAnnealingLR(
            my_optimizer,
            T_max=args.epochs,        # 一整个周期为总 epoch
            eta_min=args.lr_min if hasattr(args, 'lr_min') else 1e-6
        )

    return scheduler


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0

    for i in range(Img.shape[0]):
        #        print("Iclean[i, :, :, :] = ",Iclean[i, :, :, :].transpose(1,2,0).shape)
        PSNR += peak_signal_noise_ratio(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)

    return (PSNR / Img.shape[0])

def batch_PSNR_SSIM(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    SSIM = 0
    #    print(Img.shape,Iclean.shape)
    for i in range(Img.shape[0]):
        PSNR += peak_signal_noise_ratio(Iclean[i, :, :, :], Img[i, :, :, :], data_range=data_range)
        SSIM += ssim(Iclean[i, :, :, :].transpose(1, 2, 0),
                     Img[i, :, :, :].transpose(1, 2, 0),
                     data_range=data_range, channel_axis=2)
    return (PSNR / Img.shape[0]), (SSIM / Img.shape[0])

def range_compressor_tensor(x):
#    const_1 = torch.from_numpy(np.array(1.0)).cuda()
#    const_5000 = torch.from_numpy(np.array(5000.0)).cuda()
    const_1 = torch.tensor(1.0, device=x.device)  # 将常数放到x所在的设备
    const_5000 = torch.tensor(5000.0, device=x.device)  # 同上
    return (torch.log(const_1 + const_5000 * x)) / torch.log(const_1 + const_5000)


log_folder = "log"
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
log_file = os.path.join(log_folder, datetime.now().strftime("%Y%m%d_%H%M%S") + args.project_name + ".txt")
def log_message(message):
    """记录日志到文件"""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_entry = f"[{timestamp}] {message}\n"
    print(log_entry.strip())  # 同时输出到终端
    with open(log_file, "a") as log:
        log.write(log_entry)
############################## vimeo######################################
def hdr_to_ldr(img, expo, gamma=2.2, stdv1=1e-3, stdv2=1e-3):
    # add noise to low expo
    if expo == 1. or expo == 4.:
        stdv = np.random.rand(*img.shape) * (stdv2 - stdv1) + stdv1
        noise = np.random.normal(0, stdv)
        img = (img + noise).clip(0, 1)
    img = np.power(img * expo, 1.0 / gamma)
    img = img.clip(0, 1)
    return img
def read_list(list_path,ignore_head=False, sort=False):
    lists = []
    with open(list_path) as f:
        lists = f.read().splitlines()
    if ignore_head:
        lists = lists[1:]
    if sort:
        lists.sort(key=natural_keys)
    return lists

def random_crop(inputs, size, margin=0):
    is_list = type(inputs) == list 
    if not is_list: inputs = [inputs]

    outputs = []
    h, w, _ = inputs[0].shape
    c_h, c_w = size
    if h != c_h or w != c_w:
        t = random.randint(0+margin, h - c_h-margin)
        l = random.randint(0+margin, w - c_w-margin)
        for img in inputs:
            outputs.append(img[t: t+c_h, l: l+c_w])
    else:
        outputs = inputs
    if not is_list: outputs = outputs[0]
    return outputs

def random_flip_lrud(inputs):
    if np.random.random() > 0.5:
        return inputs
    is_list = type(inputs) == list 
    if not is_list: inputs = [inputs]

    outputs = []
    vertical_flip = True if np.random.random() > 0.5 else False # vertical flip
    for img in inputs:
        flip_img = np.fliplr(img)
        if vertical_flip:
            flip_img = np.flipud(flip_img)
        outputs.append(flip_img.copy())
    if not is_list: outputs = outputs[0]
    return outputs

