from __future__ import  absolute_import
from __future__ import  division
import torch as t
from data.voc_dataset import VOCBboxDataset
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from data import util
import numpy as np
from utils.config import opt


#逆正则化，还原为原始图片(RBG 0-255) 以便显示（可视化）的时候使用

#img维度为[[B,G,R],H,W],因为caffe预训练模型输入为BGR 0-255图片，pytorch预训练模型采用RGB 0-1图片
def inverse_normalize(img):
   #如果采用caffe预训练模型，则返回 img[::-1, :, :]  （[[B,G,R],H,W]），如果不采用，则返回(img * 0.225 + 0.45).clip(min=0, max=1) * 255 
    if opt.caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))#对应下面caffe_normalize中的减去均值
        return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255
 #pytorch_normalze中标准化为减均值除以标准差，现在乘以标准差加上均值还原回去，转换为0-255

    
   #采用pytorch预训练模型对图片预处理，函数输入的img为0-1（在下面prepare函数中变的）
def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    #'imagenet' 的均值和方差
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
       ##from torchvision import transforms as tvtsf
    img = normalize(t.from_numpy(img).float())##add .float() to avoid error:.... #(ndarray) → Tensor
    return img.numpy()

 #采用caffe预训练模型时对输入图像进行标准化，函数输入的img为0-1（在下面prepare函数中变的）
def caffe_normalize(img):
    """
    return appr -125-125 BGR
    """
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1) #转换为与img维度相同，然后下面相加
    img = (img - mean).astype(np.float32, copy=True)
    return img

#函数输入的img为0-255（原始图片）This is in CHW and RGB format.The range of its value is :math:`[0, 255]`.
def preprocess(img, min_size=600, max_size=1000):  #按照论文长边不超1000，短边不超600。按此比例缩放
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray: A preprocessed image.

    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2) #选小的比例，这样长和宽都能放缩到规定的尺寸，至少有个一是等号
    img = img / 255.  #此处转换为0-1对应正则化的输入
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
    # both the longer and shorter should be less than
    # max_size and min_size
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalze
    return normalize(img)


class Transform(object):

    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape   #_表示临时变量，一般用不到
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H      #得出bbox缩放比因子
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))#bbox按照与原图等比例缩放

        # horizontally flip
        img, params = util.random_flip(
            img, x_random=True, return_param=True)
        bbox = util.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale


class Dataset:     #训练集样本的生成
    def __init__(self, opt):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir)   #实例化 类
        self.tsf = Transform(opt.min_size, opt.max_size)

    def __getitem__(self, idx):  #__ xxx__运行Dataset类时自动运行
        ori_img, bbox, label, difficult = self.db.get_example(idx)
    #调用VOCBboxDataset中的get_example（）从数据集存储路径中将img, bbox, label, difficult 一个个的获取出来
    
        img, bbox, label, scale = self.tsf((ori_img, bbox, label))
    #调用前面的Transform函数将图片,label进行最小值最大值放缩归一化，重新调整bboxes的大小，然后随机反转，最后将数据集返回 
       
        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)


class TestDataset:    #测试集样本的生成
    def __init__(self, opt, split='test', use_difficult=True):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir, split=split, use_difficult=use_difficult)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img = preprocess(ori_img)
        return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.db)
