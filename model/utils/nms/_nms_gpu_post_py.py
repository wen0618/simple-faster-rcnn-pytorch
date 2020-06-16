#如果环境不允许或其他原因 我们没有使用Cython加速 那么将加载这个python原生函数（慢很多） 代码与_nms_gpu_post.pyx 大同小异 这里不做讲解


import numpy as np

def _nms_gpu_post( mask,
                  n_bbox,
                   threads_per_block,
                   col_blocks
                  ):
    n_selection = 0
    one_ull = np.array([1],dtype=np.uint64)
    selection = np.zeros((n_bbox,), dtype=np.int32)
    remv = np.zeros((col_blocks,), dtype=np.uint64)

    for i in range(n_bbox):
        nblock = i // threads_per_block
        inblock = i % threads_per_block

        if not (remv[nblock] & one_ull << inblock):
            selection[n_selection] = i
            n_selection += 1

            index = i * col_blocks
            for j in range(nblock, col_blocks):
                remv[j] |= mask[index + j]
    return selection, n_selection
