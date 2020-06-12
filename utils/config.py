from pprint import pprint

####配置文件，定义class Config并用opt实例化
# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # data
    voc_data_dir = '/home/cy/.chainer/dataset/pfnet/chainercv/voc/VOCdevkit/VOC2007/'
    min_size = 600  # image resize
    max_size = 1000 # image resize
    num_workers = 8
    test_num_workers = 8

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3


    # visualization
    env = 'faster-rcnn'  # visdom env
    port = 8097#visdom 端口
    plot_every = 40  # vis every N iter

    # preset
    data = 'voc'
    pretrained_model = 'vgg16'

    # training
    epoch = 14


    use_adam = False # Use Adam optimizer
    use_chainer = False # try match everything as chainer
    use_drop = False # use dropout in RoIHead
    # debug
    debug_file = '/tmp/debugf'

    test_num = 10000
    # model
    load_path = None

    caffe_pretrain = False # use caffe pretrained model instead of torchvision
    caffe_pretrain_path = 'checkpoints/vgg16_caffe.pth'

    def _parse(self, kwargs):  #解析（parse，pa：z）并设置用户设定的参数
        state_dict = self._state_dict()  #读取Config类的所有参数dict{para_name:para_value}，该函数见底部
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v) #setattr(object, name, value) 函数用于设置属性（name)的值(value)，该属性不一定是存在的。

        print('======user config========')
        pprint(self._state_dict()) #打印参数 pprint函数是为了让显示结果更加好看，，print()输出结果都在一行，
             # 不方便查看，而pprint()采用分行打印输出，所以对于数据结构比较复杂、数据长度较长的数据，适合采用pprint()打印方式
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config() #创建config对象opt
