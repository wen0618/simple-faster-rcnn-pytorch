#非极大值抑制主函数 注：cupy与numpy函数库无太大差异，np.函数 在cp上也都有，用法也一样 只不过一个在gpu计算 一个在本地计算

from __future__ import division
import numpy as np
import cupy as cp
import torch as t
try:
    from ._nms_gpu_post import _nms_gpu_post #尝试加载pyd
except:  #加载失败 告诉使用者，使用pyd会更快和生成pyd的方法
    import warnings  
    warnings.warn('''
    the python code for non_maximum_suppression is about 2x slow
    It is strongly recommended to build cython code: 
    `cd model/utils/nms/; python3 build.py build_ext --inplace''')
    from ._nms_gpu_post_py import _nms_gpu_post  #加载python原生函数

#cupy函数 这里直接使用cuda编程   kernel_name是函数名字，方便调用。code是原生c-cuda-code需要自己编写
#函数作用就是让我们自己编写的c cuda code变成一个函数 供我们调用。

@cp.util.memoize(for_each_device=True)
def _load_kernel(kernel_name, code, options=()):
    cp.cuda.runtime.free(0)
    assert isinstance(options, tuple)
    kernel_code = cp.cuda.compile_with_cache(code, options=options)
    return kernel_code.get_function(kernel_name)

#根据传入的box 和score 规定的thresh 计算返回nms处理后选择的box索引（按照分数从高到低排序）
def non_maximum_suppression(bbox, thresh, score=None,
                            limit=None):
    """Suppress bounding boxes according to their IoUs.

    This method checks each bounding box sequentially and selects the bounding
    box if the Intersection over Unions (IoUs) between the bounding box and the
    previously selected bounding boxes is less than :obj:`thresh`. This method
    is mainly used as postprocessing of object detection.
    The bounding boxes are selected from ones with higher scores.
    If :obj:`score` is not provided as an argument, the bounding box
    is ordered by its index in ascending order.

    The bounding boxes are expected to be packed into a two dimensional
    tensor of shape :math:`(R, 4)`, where :math:`R` is the number of
    bounding boxes in the image. The second axis represents attributes of
    the bounding box. They are :math:`(y_{min}, x_{min}, y_{max}, x_{max})`,
    where the four attributes are coordinates of the top left and the
    bottom right vertices.

    :obj:`score` is a float array of shape :math:`(R,)`. Each score indicates
    confidence of prediction.

    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    an input. Please note that both :obj:`bbox` and :obj:`score` need to be
    the same type.
    The type of the output is the same as the input.

    Args:
        bbox (array): Bounding boxes to be transformed. The shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
        thresh (float): Threshold of IoUs.
        score (array): An array of confidences whose shape is :math:`(R,)`.
        limit (int): The upper bound of the number of the output bounding
            boxes. If it is not specified, this method selects as many
            bounding boxes as possible.

    Returns:
        array:
        An array with indices of bounding boxes that are selected. \
        They are sorted by the scores of bounding boxes in descending \
        order. \
        The shape of this array is :math:`(K,)` and its dtype is\
        :obj:`numpy.int32`. Note that :math:`K \\leq R`.

    """

    return _non_maximum_suppression_gpu(bbox, thresh, score, limit)


def _non_maximum_suppression_gpu(bbox, thresh, score=None, limit=None):
    if len(bbox) == 0: #没有输入候选框 直接返回0
        return cp.zeros((0,), dtype=np.int32)

    n_bbox = bbox.shape[0] #传入box的个数

    if score is not None: #传入了score分数
        order = score.argsort()[::-1].astype(np.int32) #获得按分数降序排列的索引
    else:                 #没有传入
        order = cp.arange(n_bbox, dtype=np.int32)#索引等于 0-(n_box-1)  #0，1，2，...

    sorted_bbox = bbox[order, :]  #将传入的候选框 按分数从高到低排序
    selec, n_selec = _call_nms_kernel(   
        sorted_bbox, thresh)
    #调用nms_kernel函数 返回选中框的(对应于sorted_bbox的)索引 和 选中框的个数（索引是依据sorted_boxed的，也可以说是依据order的） 
    selec = selec[:n_selec]
    selec = order[selec] #根据order取 框真正的索引
    if limit is not None: #限制返回框个数 参数自己定
        selec = selec[:limit]
    return cp.asnumpy(selec)  #转化为numpy返回  返回的是选择框的索引shape:（N，）


_nms_gpu_code = '''
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0)) #进位除法 保证全线程覆盖数据
int const threadsPerBlock = sizeof(unsigned long long) * 8;

__device__ #将从gpu本地函数进行调用
inline float devIoU(float const *const bbox_a, float const *const bbox_b) { #计算两个box的IOU 
  float top = max(bbox_a[0], bbox_b[0]);   #两框重合部分上界ymin
  float bottom = min(bbox_a[2], bbox_b[2]);#两框重合部分的下界ymax
  float left = max(bbox_a[1], bbox_b[1]);  #两框重合部分的左边界xmin
  float right = min(bbox_a[3], bbox_b[3]); #两框重合部分的右边界xmax
  float height = max(bottom - top, 0.f);   #重合部分的高
  float width = max(right - left, 0.f);    #重合部分的宽
  float area_i = height * width;           #重合部分的面积
  float area_a = (bbox_a[2] - bbox_a[0]) * (bbox_a[3] - bbox_a[1]); #框a的面积
  float area_b = (bbox_b[2] - bbox_b[0]) * (bbox_b[3] - bbox_b[1]); #框b的面积
  return area_i / (area_a + area_b - area_i); #返回二者的IOU
}

extern "C"
__global__  #将从cpu调用 我们调用入口点
#下面我要举例子了因为干讲怕理解不了，我们假设n_bbox=2000
#  那么blocks=(32,32,1)  threads=(64,1,1)   我们想象blocks就是32*32的格子 每个格子有64个线程在工作 每个格子互不干扰
#  那么传入2000个框时，我们将拥有32*32*64=65536个线程在同时计算 blocks和threads可以让我们寻找到这个线程 就是下面的blockIdx.y
#  这个是cuda自行给我们标识的 我们可以直接调用获得当前线程 blockIdx=0-31  blockidy=0-31 threadIdx.x=0-63 
#  有了这个你就知道当前线程是哪个了
void nms_kernel(const int n_bbox, const float thresh,
                const float *dev_bbox,
                unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  const int row_size =
        min(n_bbox - row_start * threadsPerBlock, threadsPerBlock);#保证不越界 限制线程
  const int col_size =
        min(n_bbox - col_start * threadsPerBlock, threadsPerBlock);#保证不越界 限制线程

  __shared__ float block_bbox[threadsPerBlock * 4];#一个block的thread共享一片内存 不同block互不影响
  if (threadIdx.x < col_size) { #threadIdx.x < col_size是保证不越界，这里的目的是将box数据 复制到block_bbox
    block_bbox[threadIdx.x * 4 + 0] =
        dev_bbox[(threadsPerBlock * col_start + threadIdx.x) * 4 + 0];
    block_bbox[threadIdx.x * 4 + 1] =
        dev_bbox[(threadsPerBlock * col_start + threadIdx.x) * 4 + 1];
    block_bbox[threadIdx.x * 4 + 2] =
        dev_bbox[(threadsPerBlock * col_start + threadIdx.x) * 4 + 2];
    block_bbox[threadIdx.x * 4 + 3] =
        dev_bbox[(threadsPerBlock * col_start + threadIdx.x) * 4 + 3];
  }
  __syncthreads(); #等待同步 所有线程必须都干完了才能开始下面的工作

  if (threadIdx.x < row_size) { #保证不越界
        const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x; #框idx
    const float *cur_box = dev_bbox + cur_box_idx * 4; #生成框游标索引
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) { #跳过相同数据 自己和自己比IOU
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      if (devIoU(cur_box, block_bbox + i * 4) >= thresh) { #分别计算IOU 如果大于阈值 那么将t的i位置为1
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n_bbox, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t; #储存t值
  }
}
'''


def _call_nms_kernel(bbox, thresh):
    # PyTorch does not support unsigned long Tensor.
    # Doesn't matter,since it returns ndarray finally.
    # So I'll keep it unmodified.
    n_bbox = bbox.shape[0] #框的个数
    threads_per_block = 64  #一个block有多少thread
    col_blocks = np.ceil(n_bbox / threads_per_block).astype(np.int32)#cuda常用的对齐block操作 保证线程数最小限度全覆盖数据
    blocks = (col_blocks, col_blocks, 1)  #因为对齐一个blocks按理说是(n_blocks,1,1) 说明后面要全排列了
    threads = (threads_per_block, 1, 1)

    mask_dev = cp.zeros((n_bbox * col_blocks,), dtype=np.uint64)#开辟64*n_box*sizeof(np.uint64)的连续内存 置为0 用于存放结果
    bbox = cp.ascontiguousarray(bbox, dtype=np.float32) #将bbox从numpy转成cupycuda计算 放到连续的内存中以便cuda运算 很重要
    kern = _load_kernel('nms_kernel', _nms_gpu_code)#/加载自己写的c-cuda核函数
    kern(blocks, threads, args=(cp.int32(n_bbox), cp.float32(thresh),   #调用核函数
                                bbox, mask_dev))

    mask_host = mask_dev.get() #将计算结果从gpu取到本地
    selection, n_selec = _nms_gpu_post(
        mask_host, n_bbox, threads_per_block, col_blocks) #调用我们Cython导入的nms函数进行计算
    return selection, n_selec
