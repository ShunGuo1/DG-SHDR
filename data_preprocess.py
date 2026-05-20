import numpy as np
import cv2
import os
import math
import h5py as h5
import re

MAX_PIX_VAL_train = 255
MAX_PIX_VAL_test = 255
MAX_PIX_VAL_test16 = 65535




CROP_SIZE = 128



def read_expo(file_path):
    with open(file_path) as fp:# 打开文件路径，并以只读模式打开文件
        lines = fp.readlines()# 读取文件中的所有行，并将其存储在列表中
        assert len(lines) == 3# 断言文件中应有3行内容，若不是则会引发异常
    return pow(2, float(lines[0])), pow(2, float(lines[1])), pow(2, float(lines[2])) #将文件中的三行内容分别转换为浮点数，并对每个值进行2的幂运算，然后以元组形式返回这些结果

def get_name(count, length=6):
    s = str(count)
    l = len(s)
    for i in range(length - l):
        s = '0' + s
    return s


# def list_filter(file_list, tail):
#     r = []
#     for f in file_list:# 分割文件名和扩展名，并存储在变量 s 中
#         s = os.path.splitext(f)# 如果文件的扩展名与指定的尾部字符串匹配，则将文件加入结果列表 r
#         if s[1] == tail:
#             r.append(f)
#     return r #将后缀为tail的文件名放入r并返回
def list_filter(file_list, tail):
    r = []
    for f in file_list:
        s = os.path.splitext(f)
        if s[1] == tail:
            r.append(f)

    # 排序：提取 _exp 后的数字按数值排序
    def sort_key(filename: str):
        match = re.search(r"_exp(\d+)", filename)
        if match:
            return int(match.group(1))
        return 0

    return sorted(r, key=sort_key)
#
# def pre_process(img, val):
#     img = img.astype(np.float32)# 将图像数组转换为 np.float32 类型
#     img = img / val# 将图像数组中的每个元素除以给定的值 val
#     return img
#
#
# def pre_process_train(img):
#     img = img.astype(np.float32)
#     img = img / MAX_PIX_VAL_train
#     return img
#
#
# def pre_process_train16(img):
#     img = img.astype(np.float32)
#     img = img / MAX_PIX_VAL_test16
#     return img
#
#
# def pre_process_test(img):
#     img = img.astype(np.float32)
#     img = img / MAX_PIX_VAL_test
#     return img
#
#
# def pre_process_test16(img):
#     img = img.astype(np.float32)
#     img = img / MAX_PIX_VAL_test16
#     return img#返回图像数组


def generate_anchors(shape, size):
    def process_line(l, size):
        n = math.ceil(l / size)
        step = 128
        pos = 0
        pos_list = []
        while (pos + size < l + step):
            if (pos + size) <= l:
                pos_list.append(pos)
            else:
                pos_list.append(l - size)
                break
            pos += step
        return pos_list

    h = shape[0]
    w = shape[1]
    pos_list = []
    h_list = process_line(h, size)
    w_list = process_line(w, size)
    for i in range(len(h_list)):
        for j in range(len(w_list)):
            pos_list.append((h_list[i], w_list[j]))
    return pos_list


def load_train_scene(scene_path, h5_path, total_count):
    file_list = os.listdir(scene_path)
    if 'exposure.txt' in file_list:
        expos = read_expo(scene_path + 'exposure.txt')
    else:
        expos = read_expo(scene_path + 'exposure.txt')  # r
    # if os.path.exists(scene_path + 'Exposures.txt'):
    #     expos = read_expo(scene_path + 'Exposures.txt') #读取场景曝光信息并储存在expos中
    # else:
    #     expos = read_expo(scene_path + 'ExposureBias.txt')
    # alpha = read_txt(scene_path + 'alpha.txt')
    hdr = cv2.imread(scene_path + 'hdr_img.hdr', flags=-1)  # r
    # load ldr inputs
    input_img_list = list_filter(file_list, '.tif')#获取输入列表中后缀为tif的文件
    ldr_list = []

    assert len(input_img_list) == 3# 确保输入图像列表包含3张图片
    for i, img_path in enumerate(input_img_list):
        print(img_path)
        img = cv2.imread(scene_path + img_path, flags=-1)
        ldr_list.append(img)

    input_ldrs = np.concatenate(ldr_list, axis=-1)  #将所有输入图像连接起来形成一张大图
    # crop image into patches for training
    anchors = generate_anchors(input_ldrs.shape, CROP_SIZE) #生成锚点
    for anchor in anchors:
        y = anchor[0]
        x = anchor[1]
        # print(y)
        # print(x)
        # exit()
        _ldr = input_ldrs[y:y + CROP_SIZE, x:x + CROP_SIZE]#根据锚点裁剪输入图像和hdr图像
        _hdr = hdr[y:y + CROP_SIZE, x:x + CROP_SIZE]
        if _ldr.shape[0] != CROP_SIZE or _ldr.shape[1] != CROP_SIZE: #如果裁剪后的尺寸不符合要求，则跳过
            continue
        total_count = total_count + 1
        #创建h5文件并保存裁剪后的图像数据
        h5_file_name = os.path.join(h5_path, get_name(total_count) + '.h5')
        f = h5.File(h5_file_name, 'w')
        f['ldr'] = _ldr
        f['hdr'] = _hdr
        f['expos'] = np.array(expos)
        # f['alpha'] = np.array(alpha)

    return total_count


def load_test_scene(scene_path, h5_path, total_count):
    file_list = os.listdir(scene_path)
    # print(file_list)
    # load exposure times
    if 'ExpoBias.txt' in file_list:
        expos = read_expo(scene_path + 'ExpoBias.txt')
    else:
        expos = read_expo(scene_path + 'ExposureBias.txt')
    print(scene_path + 'ExpoBias.txt')

    # if os.path.exists(scene_path + 'Exposures.txt'):
    #     expos = read_expo(scene_path + 'Exposures.txt')
    # else:
    #     expos = read_expo(scene_path + 'ExposureBias.txt')
    # # alpha = read_txt(scene_path + 'alpha.txt')
    # hdr = cv2.imread(scene_path + 'ref_hdr_aligned.hdr', flags=-1)
    hdr = cv2.imread(scene_path + 'GT_HDR.hdr', flags=-1)
    # load ldr inputs
    input_img_list = list_filter(file_list, '.tif')
    ldr_list = []
    print(input_img_list)
    # exit()
    assert len(input_img_list) == 3
    for i, img_path in enumerate(input_img_list):
        print(img_path)
        img = cv2.imread(scene_path + img_path, flags=-1)
        ldr_list.append(img)

    # print(ldr_list)
    # exit()
    input_ldrs = np.concatenate(ldr_list, axis=-1)
    #print(input_ldrs.shape)

    total_count += 1
    h5_file_name = os.path.join(h5_path, get_name(total_count) + '.h5')
    f = h5.File(h5_file_name, 'w')
    f['ldr'] = input_ldrs
    f['hdr'] = hdr
    f['expos'] = np.array(expos)

    return total_count


def prepare_training_dataset(training_scene_path, h5_path):
    # scenes = os.listdir(training_scene_path)# 获取训练场景路径下的所有场景文件夹列表
    # # print(scenes)
    # scenes.sort()# 对场景进行排序
    # # print(scenes)
    # # exit()
    paper_scene = sorted(os.listdir(training_scene_path), key=numerical_sort)
    count = 0
    for i, scene in enumerate(paper_scene):
        if len(scene) > 0:
            print(i, 'loading scene ' + scene, count) # 打印加载场景信息
            #count = load_train_scene(training_scene_path + '/' + scene + '/reference/', h5_path, count)
            count = load_train_scene(training_scene_path + '/' + scene +'/', h5_path, count)
            # count = load_train_scene(training_scene_path + '/' + scene + '/', h5_path, count)  #调用load_train_scene 函数加载训练场景数据，并更新计数器
        else:
            continue  # 如果场景名为空则继续下一轮循环



def prepare_test_dataset(test_scene_path, h5_path):
    count = 0
    from natsort import natsorted
    paper_scene = natsorted(os.listdir(TEST_SCENE_PATH), key=numerical_sort)

    for i, scene in enumerate(paper_scene):
        print('loading paper scene ' + scene)
        # count = load_test_scene(test_scene_path + '/' + scene + '/', h5_path, count)
        count = load_test_scene(test_scene_path + '/' + scene + '/reference/', h5_path, count)
    # extra_scene = os.listdir(test_scene_path + '/' +'EXTRA/')
    # extra_scene.sort()
    # for i, scene in enumerate(extra_scene):
    #     print('loading extra scene ' + scene)
    #     count = load_test_scene(test_scene_path + '/' + 'EXTRA/' + scene + '/', h5_path, count)


def numerical_sort(value):
    numbers = re.findall(r'\d+', value)
    return int(numbers[0]) if numbers else 0

# 读取文件名并按数字顺序排序
# list_test = sorted(os.listdir(TEST_SCENE_PATH), key=numerical_sort)
# print(list_test)
# exit()
if __name__ == "__main__":
    TRAINING_SCENE_PATH = r'D:\image dataset\challenge123\train'
    TEST_SCENE_PATH = r'D:\image dataset\P19\Testing_set'
    h5_path = r'D:\image dataset\challenge123\canon_train_h5'
    os.makedirs(h5_path, exist_ok=True)
    prepare_training_dataset(TRAINING_SCENE_PATH, h5_path)#TRAINING_SCENE_PATH = D:\dataset\P19\Training_set
    # prepare_test_dataset(TEST_SCENE_PATH, h5_path)

'''
img = cv2.imread('PAPER/BarbequeDay/HDRImg.hdr', flags = cv2.IMREAD_ANYDEPTH)
print(read_expo('PAPER/BarbequeDay/exposure.txt'))
'''