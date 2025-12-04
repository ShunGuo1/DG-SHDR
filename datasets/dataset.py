import torch
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
import random
import re
from natsort import natsorted
import h5py as h5


def pre_process_png(img):
    img = img.astype(np.float32)
    img = img / 255
    return img

def pre_process_tif(img):
    img = img.astype(np.float32)
    img = img / 65535
    return img


def formulate_hdr(x):
    assert len(x.shape) == 4
    _hdr = torch.clamp(x, 0, 1)
    _hdr = torch.round(_hdr[0] * 255)
    return _hdr


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


def generate_anchors(shape,size):
    def process_line(l, size):
        math.ceil(l / size)
        step = 128  # 设置步长为128，每次移动128的像素来设置锚点
        pos = 0  # 初始位置
        pos_list = []
        while pos+size < l+step:#
            if pos+size <= l:
                pos_list.append(pos)
            else:
                pos_list.append(l-size) #将最后一个锚点设置在图像的边界内部，确保图像块在图像内部
                break
            pos += step
        return pos_list #返回锚点列表
    h = shape[0]
    w = shape[1]
    pos_list =[]
    h_list = process_line(h, size)#高度方向的锚点列表 #[0, 128, 256, 384, 512, 640]
    w_list = process_line(w, size)  #[0, 128, 256, 384, 512, 640, 768, 896, 1024, 1152]

    for i in range(len(h_list)):
        for j in range(len(w_list)):
            pos_list.append((h_list[i], w_list[j]))#二维锚点

    return pos_list


class ExposureDataset_train(Dataset):
    def __init__(self, low_exp_dir, mid_exp_dir, crop_size):
        self.low_exp_dir = low_exp_dir
        self.mid_exp_dir = mid_exp_dir
        self.low_exp_images = os.listdir(low_exp_dir)
        self.mid_exp_images = os.listdir(mid_exp_dir)
        self.crop_size = crop_size

        first_image_path = os.path.join(self.low_exp_dir, self.low_exp_images[0])
        first_image = cv2.imread(first_image_path, flags=-1)
        self.num_anchors_per_image = len(generate_anchors(first_image.shape, self.crop_size))

    def __len__(self):

        return self.num_anchors_per_image * len(self.low_exp_images)

    def __getitem__(self, idx):
        image_idx = idx // self.num_anchors_per_image
        anchor_idx = idx % self.num_anchors_per_image


        low_exp_image_path = os.path.join(self.low_exp_dir, self.low_exp_images[image_idx])
        mid_exp_image_path = os.path.join(self.mid_exp_dir, self.mid_exp_images[image_idx])

        low_exp_image = cv2.imread(low_exp_image_path, flags=-1)
        mid_exp_image = cv2.imread(mid_exp_image_path, flags=-1)

        anchors = generate_anchors(low_exp_image.shape, self.crop_size)  # 生成锚点
        y, x = anchors[anchor_idx]

        low_image = low_exp_image[y:y + self.crop_size, x:x + self.crop_size]  # 根据锚点裁剪输入图像和gt图像
        mid_image = mid_exp_image[y:y + self.crop_size, x:x + self.crop_size]

        if low_image.shape[:2] != (self.crop_size, self.crop_size):# 如果裁剪后的尺寸不符合要求，则跳过
            raise ValueError("error size")

        method = random.randint(0, 5)
        low_image = data_augmentation(low_image, method)
        mid_image = data_augmentation(mid_image, method)

        low_image = pre_process_png(low_image)
        mid_image = pre_process_png(mid_image)
        # low_image = pre_process_png(low_exp_image)
        # mid_image = pre_process_png(mid_exp_image)


        low_image = torch.as_tensor(low_image, dtype=torch.float32).permute(2, 0, 1)
        mid_image = torch.as_tensor(mid_image, dtype=torch.float32).permute(2, 0, 1)

        return low_image, mid_image



class ExposureDataset_test(Dataset):
    def __init__(self, low_exp_dir, mid_exp_dir):
        self.low_exp_dir = low_exp_dir
        self.mid_exp_dir = mid_exp_dir
        self.low_exp_images = sorted(os.listdir(low_exp_dir), key=self._extract_number)
        self.mid_exp_images = sorted(os.listdir(mid_exp_dir), key=self._extract_number)


    def __len__(self):
        return len(self.low_exp_images)

    def __getitem__(self, idx):
        low_exp_image_path = os.path.join(self.low_exp_dir, self.low_exp_images[idx])
        mid_exp_image_path = os.path.join(self.mid_exp_dir, self.mid_exp_images[idx])
        low_exp_image = cv2.imread(low_exp_image_path, flags=-1)
        mid_exp_image = cv2.imread(mid_exp_image_path, flags=-1)

        h, w = low_exp_image.shape[0], low_exp_image.shape[1]
        h = h - h % 16
        w = w - w % 16

        low_exp_image = low_exp_image[:h, :w, :]
        mid_exp_image = mid_exp_image[:h, :w, :]
        low_exp_image = pre_process_png(low_exp_image)
        mid_exp_image = pre_process_png(mid_exp_image)

        low_exp_image = torch.as_tensor(low_exp_image, dtype=torch.float32).permute(2, 0, 1)
        mid_exp_image = torch.as_tensor(mid_exp_image, dtype=torch.float32).permute(2, 0, 1)

        return low_exp_image, mid_exp_image

    def _extract_number(self, filename):
        # 提取文件名中的数字部分
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else 0


class ExposureDataset_test_noref(Dataset):
    def __init__(self, low_exp_dir):
        self.low_exp_dir = low_exp_dir

        self.low_exp_images = sorted(os.listdir(low_exp_dir), key=self._extract_number)
        #self.mid_exp_images = sorted(os.listdir(mid_exp_dir), key=self._extract_number)


    def __len__(self):
        return len(self.low_exp_images)

    def __getitem__(self, idx):
        low_exp_image_path = os.path.join(self.low_exp_dir, self.low_exp_images[idx])
        #mid_exp_image_path = os.path.join(self.mid_exp_dir, self.mid_exp_images[idx])
        low_exp_image = cv2.imread(low_exp_image_path, flags=-1)
        #mid_exp_image = cv2.imread(mid_exp_image_path, flags=-1)

        h, w = low_exp_image.shape[0], low_exp_image.shape[1]
        h = h - h % 16
        w = w - w % 16

        low_exp_image = low_exp_image[:h, :w, :]
        #mid_exp_image = mid_exp_image[:h, :w, :]
        low_exp_image = pre_process_png(low_exp_image)
        #mid_exp_image = pre_process_png(mid_exp_image)

        low_exp_image = torch.as_tensor(low_exp_image, dtype=torch.float32).permute(2, 0, 1)
        #mid_exp_image = torch.as_tensor(mid_exp_image, dtype=torch.float32).permute(2, 0, 1)

        return low_exp_image #, mid_exp_image

    def _extract_number(self, filename):
        # 提取文件名中的数字部分
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else 0


# 计算对比度、饱和度和亮度的权重
def calculate_weights(img, alpha_c=1, alpha_s=1, alpha_e=1):#可调整参数
    # 计算对比度：使用梯度的绝对值作为对比度
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    contrast = cv2.Laplacian(gray, cv2.CV_64F).var()  # 计算灰度图的拉普拉斯变换

    # 计算饱和度：计算RGB通道的标准差
    mean = np.mean(img, axis=2, keepdims=True)
    saturation = np.sqrt(np.mean((img - mean) ** 2, axis=2))  # 按公式计算三通道标准差

    # 计算亮度：使用高斯函数衡量像素亮度
    img_normalized = img / 255.0  # 将像素值归一化为 [0, 1] 范围
    sigma = 0.5  # 假定方差为 0.2
    exposure = np.exp(-((img_normalized - 0.5) ** 2) / (2 * sigma ** 2)).prod(axis=2)  # 亮度权重

    # 计算最终权重并防止零除
    weight = (contrast ** alpha_c) * (saturation ** alpha_s) * (exposure ** alpha_e)
    return weight + 1e-14


# 归一化权重
def normalize_weights(weights):
    return weights / np.sum(weights, axis=0)


# 利用权重融合图片
def fuse_images(images, weights):
    fused_image = np.sum(weights[..., np.newaxis] * images, axis=0)
    return np.clip(fused_image, 0, 255).astype(np.uint8)


class y_color(Dataset):
    def __init__(self, low_exp_dir, x_mid_dir, mid_exp_dir, crop_size):#(low ,mid ,truth mid )
        self.low_exp_dir = low_exp_dir
        self.mid_exp_dir = mid_exp_dir
        self.x_mid_dir = x_mid_dir
        self.low_exp_images = os.listdir(low_exp_dir)
        self.mid_exp_images = os.listdir(mid_exp_dir)
        self.x_mid_images = os.listdir(x_mid_dir)

        self.crop_size = crop_size

        first_image_path = os.path.join(self.low_exp_dir, self.low_exp_images[0])
        first_image = cv2.imread(first_image_path, flags=-1)
        self.num_anchors_per_image = len(generate_anchors(first_image.shape, self.crop_size))

    def __len__(self):
        return self.num_anchors_per_image * len(self.low_exp_images)

    def __getitem__(self, idx):
        image_idx = idx // self.num_anchors_per_image
        anchor_idx = idx % self.num_anchors_per_image

        low_exp_image_path = os.path.join(self.low_exp_dir, self.low_exp_images[image_idx])
        mid_exp_image_path = os.path.join(self.mid_exp_dir, self.mid_exp_images[image_idx])
        x_mid_image_path = os.path.join(self.x_mid_dir, self.x_mid_images[image_idx])

        low_exp_image = cv2.imread(low_exp_image_path, flags=-1)
        mid_exp_image = cv2.imread(mid_exp_image_path, flags=-1)
        x_mid_image = cv2.imread(x_mid_image_path, flags=-1)

        anchors = generate_anchors(low_exp_image.shape, self.crop_size)  # 生成锚点
        y, x = anchors[anchor_idx]

        low_image = low_exp_image[y:y + self.crop_size, x:x + self.crop_size]  # 根据锚点裁剪输入图像和gt图像
        mid_image = mid_exp_image[y:y + self.crop_size, x:x + self.crop_size]
        x_mid_image = x_mid_image[y:y + self.crop_size, x:x + self.crop_size]

        if low_image.shape[:2] != (self.crop_size, self.crop_size):  # 如果裁剪后的尺寸不符合要求，则跳过
            raise ValueError("error size")

        method = random.randint(0, 5)
        low_image = data_augmentation(low_image, method)
        mid_image = data_augmentation(mid_image, method)
        x_mid_image = data_augmentation(x_mid_image, method)


        # 堆叠图片以计算权重
        images = np.stack([low_image, x_mid_image])

        low_image = pre_process_png(low_image)
        mid_image = pre_process_png(mid_image)
        x_mid_image = pre_process_png(x_mid_image)

        low_image = torch.as_tensor(low_image, dtype=torch.float32).permute(2, 0, 1)
        mid_image = torch.as_tensor(mid_image, dtype=torch.float32).permute(2, 0, 1)# 真实中

        # 中
        x_mid_image = torch.as_tensor(x_mid_image, dtype=torch.float32).permute(2, 0, 1)

        # 计算并归一化权重
        weights = np.array([calculate_weights(img) for img in images])
        normalized_weights = normalize_weights(weights)

        # 根据归一化权重融合图片
        O = fuse_images(images, normalized_weights) #低 中 图片与对应权重相乘后相加

        # 生成伪标签
        lambda_value = 0.5  # 可调整参数
        O= torch.as_tensor(O, dtype=torch.float32).permute(2, 0, 1)
        y_color = lambda_value * O + (1 - lambda_value) * x_mid_image

        return low_image, x_mid_image, mid_image, y_color #  低 ,中 , 真实低到中 ,y_color



def generate_anchor(shape, size):
    """
    根据图像的尺寸动态生成锚点，确保覆盖整个图像。
    """

    def process_line(l, size):
        step = 128  # 设置步长
        pos_list = []
        pos = 0
        while pos + size <= l:  # 锚点位于图像内
            pos_list.append(pos)
            pos += step
        if pos + size > l and l - size >= 0:  # 添加覆盖边缘的锚点
            pos_list.append(l - size)
        return pos_list

    h, w = shape[:2]
    h_anchors = process_line(h, size)  # 高度方向锚点
    w_anchors = process_line(w, size)  # 宽度方向锚点

    return [(y, x) for y in h_anchors for x in w_anchors]  # 笛卡尔积生成二维锚点


class transformer_train(Dataset):
    def __init__(self, low_exp_dir, mid_exp_dir, crop_size):
        """
        初始化数据集类
        :param low_exp_dir: 低曝光图片文件夹路径
        :param mid_exp_dir: 中曝光图片文件夹路径
        :param crop_size: 裁剪的大小
        """
        self.low_exp_dir = low_exp_dir
        self.mid_exp_dir = mid_exp_dir
        self.low_exp_images = sorted(os.listdir(low_exp_dir))
        self.mid_exp_images = sorted(os.listdir(mid_exp_dir))
        self.crop_size = crop_size

        # 检查两组图片数量是否一致
        assert len(self.low_exp_images) == len(self.mid_exp_images), \
            "Mismatch in number of low and mid exposure images."

    def __len__(self):
        """
        返回数据集大小
        """
        total_crops = 0
        for img_name in self.low_exp_images:
            img_path = os.path.join(self.low_exp_dir, img_name)
            img = cv2.imread(img_path, flags=-1)
            anchors = generate_anchor(img.shape, self.crop_size)
            total_crops += len(anchors)
        return total_crops

    def __getitem__(self, idx):
        """
        获取数据集中一个样本
        :param idx: 索引
        """
        # 逐图处理，根据每张图的尺寸动态生成锚点
        current_idx = idx
        for img_idx, img_name in enumerate(self.low_exp_images):
            img_path = os.path.join(self.low_exp_dir, img_name)
            img = cv2.imread(img_path, flags=-1)

            # 生成锚点
            anchors = generate_anchor(img.shape, self.crop_size)
            num_anchors = len(anchors)

            if current_idx < num_anchors:
                anchor_coords = anchors[current_idx]
                break
            current_idx -= num_anchors

        # 获取当前图像的低曝光和中曝光路径
        low_exp_image_path = os.path.join(self.low_exp_dir, self.low_exp_images[img_idx])
        mid_exp_image_path = os.path.join(self.mid_exp_dir, self.mid_exp_images[img_idx])

        # 读取图像
        low_exp_image = cv2.imread(low_exp_image_path, flags=-1)
        mid_exp_image = cv2.imread(mid_exp_image_path, flags=-1)

        # 获取锚点坐标
        y, x = anchor_coords

        # 裁剪图像
        low_image = low_exp_image[y:y + self.crop_size, x:x + self.crop_size]
        mid_image = mid_exp_image[y:y + self.crop_size, x:x + self.crop_size]

        # 确保裁剪后的尺寸符合要求
        if low_image.shape[:2] != (self.crop_size, self.crop_size):
            raise ValueError(f"Crop size mismatch at image {img_idx}, anchor {anchor_coords}")

        # 数据增强
        method = random.randint(0, 5)
        low_image = data_augmentation(low_image, method)
        mid_image = data_augmentation(mid_image, method)

        # 预处理
        low_image = pre_process_tif(low_image)
        mid_image = (np.log(1 + 5000 * mid_image)) / np.log(1 + 5000)
        # mid_image = pre_process_png(mid_image)  # 如果需要对 mid_image 进行处理，解开注释

        # 转换为张量
        low_image = torch.as_tensor(low_image.copy(), dtype=torch.float32).permute(2, 0, 1)
        mid_image = torch.as_tensor(mid_image.copy(), dtype=torch.float32).permute(2, 0, 1)

        return low_image, mid_image


#
#
# class transformer_train(Dataset):
#     def __init__(self, low_exp_dir, mid_exp_dir, crop_size):
#         self.low_exp_dir = low_exp_dir
#         self.mid_exp_dir = mid_exp_dir
#         self.low_exp_images = os.listdir(low_exp_dir)
#         self.mid_exp_images = os.listdir(mid_exp_dir)
#         self.crop_size = crop_size
#
#         first_image_path = os.path.join(self.low_exp_dir, self.low_exp_images[0])
#         first_image = cv2.imread(first_image_path, flags=-1)
#         self.num_anchors_per_image = len(generate_anchors(first_image.shape, self.crop_size))
#
#     def __len__(self):
#
#         return self.num_anchors_per_image * len(self.low_exp_images)
#
#     def __getitem__(self, idx):
#         image_idx = idx // self.num_anchors_per_image
#         anchor_idx = idx % self.num_anchors_per_image
#
#
#         low_exp_image_path = os.path.join(self.low_exp_dir, self.low_exp_images[image_idx])
#         mid_exp_image_path = os.path.join(self.mid_exp_dir, self.mid_exp_images[image_idx])
#
#         low_exp_image = cv2.imread(low_exp_image_path, flags=-1)
#         mid_exp_image = cv2.imread(mid_exp_image_path, flags=-1)
#
#         anchors = generate_anchors(low_exp_image.shape, self.crop_size)  # 生成锚点
#         y, x = anchors[anchor_idx]
#
#         low_image = low_exp_image[y:y + self.crop_size, x:x + self.crop_size]  # 根据锚点裁剪输入图像和gt图像
#         mid_image = mid_exp_image[y:y + self.crop_size, x:x + self.crop_size]
#
#         if low_image.shape[:2] != (self.crop_size, self.crop_size):# 如果裁剪后的尺寸不符合要求，则跳过
#             raise ValueError("error size")
#
#         method = random.randint(0, 5)
#         low_image = data_augmentation(low_image, method)
#         mid_image = data_augmentation(mid_image, method)
#
#         low_image = pre_process_tif(low_image)
#         #mid_image = pre_process_png(mid_image)
#
#         low_image = torch.as_tensor(low_image, dtype=torch.float32).permute(2, 0, 1)
#         mid_image = torch.as_tensor(mid_image, dtype=torch.float32).permute(2, 0, 1)
#
#         return low_image, mid_image


class transformer_test(Dataset):
    def __init__(self, low_exp_dir, mid_exp_dir):
        self.low_exp_dir = low_exp_dir
        self.mid_exp_dir = mid_exp_dir
        self.low_exp_images = sorted(os.listdir(low_exp_dir), key=self._extract_number)
        self.mid_exp_images = sorted(os.listdir(mid_exp_dir), key=self._extract_number)

    def __len__(self):
        return len(self.low_exp_images)

    def __getitem__(self, idx):
        low_exp_image_path = os.path.join(self.low_exp_dir, self.low_exp_images[idx])
        mid_exp_image_path = os.path.join(self.mid_exp_dir, self.mid_exp_images[idx])
        low_exp_image = cv2.imread(low_exp_image_path, flags=-1)
        mid_exp_image = cv2.imread(mid_exp_image_path, flags=-1)

        h, w = low_exp_image.shape[0], low_exp_image.shape[1]
        h = h - h % 16
        w = w - w % 16

        low_exp_image = low_exp_image[:h, :w, :]
        mid_exp_image = mid_exp_image[:h, :w, :]

        low_exp_image = pre_process_tif(low_exp_image)
        mid_exp_image = (np.log(1 + 5000 * mid_exp_image)) / np.log(1 + 5000)

        #mid_exp_image = pre_process_png(mid_exp_image)

        low_exp_image = torch.as_tensor(low_exp_image, dtype=torch.float32).permute(2, 0, 1)
        mid_exp_image = torch.as_tensor(mid_exp_image, dtype=torch.float32).permute(2, 0, 1)

        return low_exp_image, mid_exp_image

    def _extract_number(self, filename):
        # 提取文件名中的数字部分
        match = re.search(r'\d+', filename)
        return int(match.group()) if match else 0


class transformer_train_h5(Dataset):
    def __init__(self, path):
        super(transformer_train_h5, self).__init__()
        self.path = path
        self.h5_file_path = natsorted(os.listdir(self.path))
        self.num = len(self.h5_file_path)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        h5_single_path = self.h5_file_path[idx]
        h5_file = h5.File(self.path +'/'+h5_single_path, mode='r')

        mid = h5_file['mid'][:, :, :][()]
        gt = h5_file['gt'][:, :, :][()]
        mid_image = mid[:, :, 0:3]
        gt_image = gt[:, :, 0:3]

        mid_image = pre_process_tif(mid_image)
        gt_image = (np.log(1 + 5000 * gt_image)) / np.log(1 + 5000)

        mid_image= torch.as_tensor(mid_image, dtype=torch.float32).permute(2, 0, 1)
        gt_image = torch.as_tensor(gt_image, dtype=torch.float32).permute(2, 0, 1)

        sample = {'mid_image': mid_image, 'gt_image': gt_image}

        return sample


class transformer_test_h5(Dataset):
    def __init__(self, path):
        super(transformer_test_h5, self).__init__()
        self.path = path
        self.h5_file_path = natsorted(os.listdir(self.path))
        self.num = len(self.h5_file_path)


    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        h5_single_path = self.h5_file_path[idx]
        h5_file = h5.File(self.path +'/'+h5_single_path, mode='r')

        mid = h5_file['mid'][:, :, :][()]
        gt = h5_file['gt'][:, :, :][()]
        num = int(h5_file['num'][()])
        shape = h5_file['shape'][()]

        mid_image = mid[:, :, 0:3]
        gt_image = gt[:, :, 0:3]

        mid_image = pre_process_tif(mid_image)
        gt_image = (np.log(1 + 5000 * gt_image)) / np.log(1 + 5000)

        mid_image= torch.as_tensor(mid_image, dtype=torch.float32).permute(2, 0, 1)
        gt_image = torch.as_tensor(gt_image, dtype=torch.float32).permute(2, 0, 1)

        sample = {'mid_image': mid_image, 'gt_image': gt_image, 'num': num, 'shape': shape}

        return sample


class unet_p19_traindataset(Dataset):
    def __init__(self, train_dataset_path, crop_size=256):
        super(unet_p19_traindataset, self).__init__()
        self.train_dataset_path = train_dataset_path
        self.images_list = self.load_image()
        self.crop_size = crop_size
        self.num_anchor_per_img = len(generate_anchors(self.images_list[0].shape, self.crop_size))
        print(self.num_anchor_per_img)

    def load_image(self):
        images_list = []
        for sub_dir in natsorted(os.listdir(self.train_dataset_path)):
            sub_dir_path = os.path.join(self.train_dataset_path, sub_dir)
            #print(sub_dir_path)
            if os.path.isdir(sub_dir_path):
                reference_dir = os.path.join(str(sub_dir_path), 'reference')
                hdr = cv2.imread(reference_dir+'/'+'GT_HDR.hdr', flags=-1)
                if hdr is None:
                    raise ValueError(f"Failed to read image")
                images_list.append(hdr)
                file_names = os.listdir(reference_dir)
                input_images = list_filter(file_names, '.tif')
                for i, input_image in enumerate(input_images):
                    img = cv2.imread(reference_dir +'/'+ input_image, flags=-1)
                    images_list.append(img)

        return images_list

    def __len__(self):
        return self.num_anchor_per_img

    def __getitem__(self, index):
        gt_image = self.images_list[0]
        gt_image = (np.log(1 + 5000 * gt_image)) / np.log(1 + 5000)
        low = self.images_list[1]
        mid = self.images_list[2]
        high = self.images_list[3]

        anchors = generate_anchors(low.shape, self.crop_size)  # 生成锚点

        y, x = anchors[index]
        if y + self.crop_size > low.shape[0] or x + self.crop_size > low.shape[1]:
            raise ValueError(f"Anchor position ({y}, {x}) exceeds image dimensions for crop size {self.crop_size}.")

        low = low[y:y + self.crop_size, x:x + self.crop_size]  # 根据锚点裁剪输入图像和gt图像
        mid = mid[y:y + self.crop_size, x:x + self.crop_size]
        high = high[y:y + self.crop_size, x:x + self.crop_size]
        gt_image = gt_image[y:y + self.crop_size, x:x + self.crop_size]

        if low.shape[:2] != (self.crop_size, self.crop_size):# 如果裁剪后的尺寸不符合要求，则跳过
            raise ValueError("error size")

        # method = random.randint(0, 5)
        # low = data_augmentation(low, method)
        # mid = data_augmentation(mid, method)
        # high = data_augmentation(high, method)
        # gt_image = data_augmentation(gt_image, method)

        low = pre_process_tif(low)
        mid = pre_process_tif(mid)
        high = pre_process_tif(high)

        low = torch.as_tensor(low, dtype=torch.float32).permute(2, 0, 1)
        mid = torch.as_tensor(mid, dtype=torch.float32).permute(2, 0, 1)
        high = torch.as_tensor(high, dtype=torch.float32).permute(2, 0, 1)
        gt_image = torch.as_tensor(gt_image, dtype=torch.float32).permute(2, 0, 1)

        sample = {'low_image': low, 'mid_image': mid, 'high_image': high, 'gt_image': gt_image}

        return sample


class unet_p19_testdataset(Dataset):
    def __init__(self, test_dataset_path):
        super(unet_p19_testdataset, self).__init__()
        self.test_dataset_path = test_dataset_path
        self.images_list = self.load_image()

    def load_image(self):
        images_list = []
        for sub_dir in natsorted(os.listdir(self.test_dataset_path)):
            sub_dir_path = os.path.join(self.test_dataset_path, sub_dir)
            # print(sub_dir_path)
            if os.path.isdir(sub_dir_path):
                reference_dir = os.path.join(str(sub_dir_path), 'reference')
                hdr = cv2.imread(reference_dir + '/' + 'GT_HDR.hdr', flags=-1)
                images_list.append(hdr)
                file_names = os.listdir(reference_dir)
                input_images = list_filter(file_names, '.tif')
                for i, input_image in enumerate(input_images):
                    img = cv2.imread(reference_dir + '/' + input_image, flags=-1)
                    #print(reference_dir + '/' + input_image)
                    images_list.append(img)

        return images_list

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        gt_image = self.images_list[0]
        gt_image = (np.log(1 + 5000 * gt_image)) / np.log(1 + 5000)
        low = self.images_list[1]
        mid = self.images_list[2]
        high = self.images_list[3]

        low = pre_process_tif(low)
        mid = pre_process_tif(mid)
        high = pre_process_tif(high)

        low = torch.as_tensor(low, dtype=torch.float32).permute(2, 0, 1)
        mid = torch.as_tensor(mid, dtype=torch.float32).permute(2, 0, 1)
        high = torch.as_tensor(high, dtype=torch.float32).permute(2, 0, 1)
        gt_image = torch.as_tensor(gt_image, dtype=torch.float32).permute(2, 0, 1)

        sample = {'low_image': low, 'mid_image': mid, 'high_image': high, 'gt_image': gt_image}

        return sample


class Dataset_P19_h5(Dataset):
    def __init__(self, path):
        super(Dataset_P19_h5, self).__init__()
        self.path = path
        self.file_h5_lists = os.listdir(self.path)
        self.num = len(self.file_h5_lists)

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        single_file = self.file_h5_lists[item]
        h5_file = h5.File(self.path + single_file, mode='r')
        # _x = random.randint(0, 320- 256)###
        # _y = random.randint(0, 320- 256)
        _ldr = h5_file['ldr'][0:256, 0: 256][()]
        _hdr = h5_file['hdr'][0:256, 0: 256][()]
        #  print("befor=",_ldr.max())
        expos = h5_file['expos'][()]
        expos_ = expos / expos[1]
        expos_ = np.log2(expos_)
        expos_ = expos_ / 10

        method = random.randint(0, 5)
        _ldr = data_augmentation(_ldr, method)
        _hdr = data_augmentation(_hdr, method)
        _ldr = pre_process_tif(_ldr)

        _low = _ldr[:, :, 0:3]
        _mid = _ldr[:, :, 3:6]
        _high = _ldr[:, :, 6:9]

        _x1, _x2, _x3 = _low.transpose(2, 0, 1), _mid.transpose(2, 0, 1), _high.transpose(2, 0, 1)
        _hdr = _hdr.transpose(2, 0, 1)
        _hdr = (np.log(1 + 5000 * _hdr)) / np.log(1 + 5000)

        sample = {'low_image': _x1.copy(), 'mid_image': _x2.copy(), 'high_image': _x3.copy(), 'hdr_gt': _hdr.copy(),
                  'expos': expos.copy(), 'expos_att': expos_.copy()}

        return sample


class Dataset_h5_P19_test(Dataset):
    def __init__(self, path):
        super(Dataset_h5_P19_test, self).__init__()
        self.path = path
        self.file_h5_lists = natsorted(os.listdir(self.path))
        self.num = len(self.file_h5_lists)

    def __getitem__(self, item):
        single_file = self.file_h5_lists[item]
        h5_file = h5.File(self.path + single_file, mode='r')

        _ldr = h5_file['ldr'][:, :, :][()]
        _hdr = h5_file['hdr'][:, :, :][()]##
        expos = h5_file['expos'][()]
        # if _ldr.shape[0] > _ldr.shape[1]:
        #     _ldr = _ldr.transpose(1, 0, 2)
        #     _hdr = _hdr.transpose(1, 0, 2)  ##
        _ldr = _ldr[:, :, :]
        _hdr = _hdr[:, :, :]##

        # expos = h5_file['expos'][()]
        expos_ = expos / expos[1]
        expos_ = np.log2(expos_)
        expos_ = expos_ / 10

        _ldr = pre_process_tif(_ldr)
        # _hdr = pre_process_png(_hdr)
        # _hdr = pre_process_png(_hdr)
        # p19
        _x1 = _ldr[:, :, 0:3] # 低
        _x2 = _ldr[:, :, 3:6] # 中
        _x3 = _ldr[:, :, 6:9] # 高

        _x1, _x2, _x3 = _x1.transpose(2, 0, 1), _x2.transpose(2, 0, 1), _x3.transpose(2, 0, 1)
        _hdr = _hdr.transpose(2, 0, 1)#
        #_hdr = (np.log(1 + 5000 * _hdr)) / np.log(1 + 5000)

        sample = {'low_image': _x1.copy(), 'mid_image': _x2.copy(), 'high_image': _x3.copy(), 'hdr_gt': _hdr.copy(),
                  'expos': expos.copy(), 'expos_att': expos_.copy()}

        return sample

    def __len__(self):
        return self.num
