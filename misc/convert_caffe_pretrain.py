# code from ruotian luo
# https://github.com/ruotianluo/pytorch-faster-rcnn
import torch
from torch.utils.model_zoo import load_url
from torchvision import models

#下载预训练的caffe_pretrain model最后训练效果mAP会高一点，所以我们需要提前运行这个python类
#下载预训练的caffe模型参数 如果不提前下载 训练时会自动用pytorch自带的vgg16模型参数 效果稍微差一点

sd = load_url("https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth") #下载并加载模型
sd['classifier.0.weight'] = sd['classifier.1.weight']
sd['classifier.0.bias'] = sd['classifier.1.bias']
del sd['classifier.1.weight']
del sd['classifier.1.bias']

sd['classifier.3.weight'] = sd['classifier.4.weight']
sd['classifier.3.bias'] = sd['classifier.4.bias']
del sd['classifier.4.weight']
del sd['classifier.4.bias']

import  os
# speicify the path to save
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')
torch.save(sd, "checkpoints/vgg16_caffe.pth")

#caffe版参数名字与pytorch的vgg16参数名字不一样 所以我们这里需要将参数改名字（因为我们要加载pytorch自带的网络结构 所以参数名称必须与其一致）
