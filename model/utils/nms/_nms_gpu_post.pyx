#通过non_maximum_suppression.py我们得到了每2个框的IOU值 并巧妙的储存了起来，这里就是选择框的过程了，
#由于我们巧妙的储存了t，所以这里也会巧妙的把t读取来，完成NMS操作。
#这里的代码纯属技法问题，如果你熟悉cuda寻址，熟悉一些算法，这里很容易还原算法。
#这个类的作用就是NMS的作用：选出score最大的框,这个框肯定框柱了一个物体，
#如果其他与这个IOU大于阀值，那么认为其他框也是框的这个物体，但是score置信度还不行，果断扔掉。
#再找一个score次高的，循环做下去…就是nms了


cimport numpy as np
from libc.stdint cimport uint64_t

import numpy as np
#根据我们的mask_dev 选出候选框 Cython写法 与python没什么不同 
#声明一些变量类型而已 Cpython会自动帮我们依据半python代码(pyd)将此pyd编译成.c c编译器会再次编译.c文件生成我们要的动态链接库
def _nms_gpu_post(np.ndarray[np.uint64_t, ndim=1] mask,
                  int n_bbox,
                  int threads_per_block,
                  int col_blocks
                  ):
    cdef:
        int i, j, nblock, index
        uint64_t inblock
        int n_selection = 0
        uint64_t one_ull = 1
        np.ndarray[np.int32_t, ndim=1] selection
        np.ndarray[np.uint64_t, ndim=1] remv

    selection = np.zeros((n_bbox,), dtype=np.int32)
    remv = np.zeros((col_blocks,), dtype=np.uint64)

    for i in range(n_bbox):#遍历2000个框
        nblock = i // threads_per_block #nblock 0-31
        inblock = i % threads_per_block #inblock 0-63 

        if not (remv[nblock] & one_ull << inblock): #如果IOU不小于阀值(标注的是一个新的物体)
            selection[n_selection] = i #记录这个box的index
            n_selection += 1  #选取框数目+1

            index = i * col_blocks  #index寻址
            for j in range(nblock, col_blocks):
                remv[j] |= mask[index + j] #将对应选中框的所有IOU对应结果存入
    return selection, n_selection
###由于我们传入的box都是按score排序好的，所以该算法可以实现。
#首先我们会进入if not语句中，记录第一个score最大的框，下面for j的循环就是找出所有与第一个框的IOU记录
#赋值给remv 下面就是依次判断第二个框第三个框是否与第一个框IOU大于阀值 移一位就是一个框的IOU记录
#如果大于阀值则继续下一个for i循环 小于阀值认定是不同的框，由于其分数最高 所以记录在selection中 再找出与此框的所有IOU记录继续遍历…
