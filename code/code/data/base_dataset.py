"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import cv2

class BaseDataset(data.Dataset, ABC):

    """该类是所有数据集类的抽象基类。
       子类需实现以下方法：
       - __init__
       - __len__
       - __getitem__
       - modify_commandline_options（可选）
       """

    def __init__(self, opt):
        """初始化类，保存参数设置。

        参数:
            opt -- 包含实验配置的选项对象，需为 BaseOptions 的子类
        """


        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):

        """可添加或修改数据集特定的命令行参数。

         参数:
             parser -- 命令行解析器
             is_train -- 是否为训练模式
         返回:
             修改后的 parser
         """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):

        """根据索引返回数据和元信息。

          参数:
              index -- 数据索引
          返回:
              包含数据和元数据的字典
          """
        pass


def get_params(opt, size):
    w, h = size  # 图像宽高
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':       # 调整为固定大小再裁剪
        new_h = new_w = opt.load_size
    elif opt.preprocess == 'scale_width_and_crop':  # 先按宽度缩放，再裁剪
        new_w = opt.load_size
        new_h = opt.load_size * h // w
    # 随机裁剪位置
    x = random.randint(0, np.maximum(0, new_w - opt.crop_size)) # 随机决定是否水平翻转
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size)) # 随机决定是否水平翻转

    flip = random.random() > 0.5

    return {'crop_pos': (x, y), 'flip': flip}

def get_transform(opt, params=None, grayscale=False, method=transforms.InterpolationMode.BICUBIC, convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.load_size, opt.crop_size, method)))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess == 'none':
        # transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))
        transform_list = transform_list

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    if convert:
        transform_list += [transforms.ToTensor()]
        
    return transforms.Compose(transform_list)


def __transforms2pil_resize(method):
    mapper = {transforms.InterpolationMode.BILINEAR: Image.BILINEAR,
              transforms.InterpolationMode.BICUBIC: Image.BICUBIC,
              transforms.InterpolationMode.NEAREST: Image.NEAREST,
              transforms.InterpolationMode.LANCZOS: Image.LANCZOS,}
    return mapper[method]

# 将图像尺寸调整为指定基数的倍数（如4）
def __make_power_2(img, base, method=transforms.InterpolationMode.BICUBIC):
    method = __transforms2pil_resize(method)
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __scale_width(img, target_size, crop_size, method=transforms.InterpolationMode.BICUBIC):
    method = __transforms2pil_resize(method)
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    """Print warning information about image size(only print once)"""
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
