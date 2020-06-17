import numpy as np
import cupy as cp

from model.utils.bbox_tools import bbox2loc, bbox_iou, loc2bbox
from model.utils.nms import non_maximum_suppression


class ProposalTargetCreator(object):
#从ProposalCreator生成的Rois中 这里是选取正负样本共n_sample个（这里是128）用于训练ROIHead（该模块类仅用于训练阶段）
#为了后续放入RoiPooling后卷积全连接进行（20+1）类的分类损失计算FC21和21x4的box位置损失计算FC84 我们规定与真实gt_box框 IOU>0.5为正样本
#0.5<IOU<0.1为负样本

    
    """Assign ground truth bounding boxes to given RoIs.

    The :meth:`__call__` of this class generates training targets
    for each object proposal.
    This is used to train Faster RCNN [#]_.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        n_sample (int): The number of sampled regions.
        pos_ratio (float): Fraction of regions that is labeled as a
            foreground.
        pos_iou_thresh (float): IoU threshold for a RoI to be considered as a
            foreground.
        neg_iou_thresh_hi (float): RoI is considered to be the background
            if IoU is in
            [:obj:`neg_iou_thresh_hi`, :obj:`neg_iou_thresh_hi`).
        neg_iou_thresh_lo (float): See above.

    """

    def __init__(self,
                 n_sample=128,
                 pos_ratio=0.25, pos_iou_thresh=0.5,
                 neg_iou_thresh_hi=0.5, neg_iou_thresh_lo=0.0
                 ):
        self.n_sample = n_sample #要选取的样本总数
        self.pos_ratio = pos_ratio #正样本率
        self.pos_iou_thresh = pos_iou_thresh#正样本iou阀值
        self.neg_iou_thresh_hi = neg_iou_thresh_hi#负样本iou最高阀值
        self.neg_iou_thresh_lo = neg_iou_thresh_lo  # NOTE:default 0.1 in py-faster-rcnn #负样本iou最低阀值

    def __call__(self, roi, bbox, label,#输入 N个ROI （N，4）   真实标注的bbox（R,4）,真实的label（R，）
                 loc_normalize_mean=(0., 0., 0., 0.),#loc均值
                 loc_normalize_std=(0.1, 0.1, 0.2, 0.2)):#loc标准差
        """Assigns ground truth to sampled proposals.

        This function samples total of :obj:`self.n_sample` RoIs
        from the combination of :obj:`roi` and :obj:`bbox`.
        The RoIs are assigned with the ground truth class labels as well as
        bounding box offsets and scales to match the ground truth bounding
        boxes. As many as :obj:`pos_ratio * self.n_sample` RoIs are
        sampled as foregrounds.

        Offsets and scales of bounding boxes are calculated using
        :func:`model.utils.bbox_tools.bbox2loc`.
        Also, types of input arrays and output arrays are same.

        Here are notations.

        * :math:`S` is the total number of sampled RoIs, which equals \
            :obj:`self.n_sample`.
        * :math:`L` is number of object classes possibly including the \
            background.

        Args:
            roi (array): Region of Interests (RoIs) from which we sample.
                Its shape is :math:`(R, 4)`
            bbox (array): The coordinates of ground truth bounding boxes.
                Its shape is :math:`(R', 4)`.
            label (array): Ground truth bounding box labels. Its shape
                is :math:`(R',)`. Its range is :math:`[0, L - 1]`, where
                :math:`L` is the number of foreground classes.
            loc_normalize_mean (tuple of four floats): Mean values to normalize
                coordinates of bouding boxes.
            loc_normalize_std (tupler of four floats): Standard deviation of
                the coordinates of bounding boxes.

        Returns:
            (array, array, array):

            * **sample_roi**: Regions of interests that are sampled. \
                Its shape is :math:`(S, 4)`.
            * **gt_roi_loc**: Offsets and scales to match \
                the sampled RoIs to the ground truth bounding boxes. \
                Its shape is :math:`(S, 4)`.
            * **gt_roi_label**: Labels assigned to sampled RoIs. Its shape is \
                :math:`(S,)`. Its range is :math:`[0, L]`. The label with \
                value 0 is the background.

        """
        n_bbox, _ = bbox.shape #真实标注的bbox个数

        roi = np.concatenate((roi, bbox), axis=0)    #（N，4）to (N+R,4），R是真实框个数 N是ROI个数

        pos_roi_per_image = np.round(self.n_sample * self.pos_ratio)     #正样本数 np.round()四舍五入取整
        iou = bbox_iou(roi, bbox)       #计算两者的IOU
        gt_assignment = iou.argmax(axis=1 ) #最大值索引，gt_assighment[i]=j  表示第i个roi 与第j个bbox的IOU最大
        max_iou = iou.max(axis=1)   #/对应上面索引的最大值
        # Offset range of classes from [0, n_fg_class - 1] to [1, n_fg_class].
        # The label with value 0 is the background.
        gt_roi_label = label[gt_assignment] + 1 #所有roi对应IOU最大的真实gt_box的label   一一对应

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0] #选择IOU大于阀值的索引
        pos_roi_per_this_image = int(min(pos_roi_per_image, pos_index.size)) #如果得到索引个数大于设定 则变为设定个数
        if pos_index.size > 0:
            pos_index = np.random.choice(
                pos_index, size=pos_roi_per_this_image, replace=False)#随机选取pos_roi_per_this_image个 

        # Select background RoIs as those within
        # [neg_iou_thresh_lo, neg_iou_thresh_hi).
        neg_index = np.where((max_iou < self.neg_iou_thresh_hi) &
                             (max_iou >= self.neg_iou_thresh_lo))[0] #IOU小于设定的索引 为负样本
        neg_roi_per_this_image = self.n_sample - pos_roi_per_this_image #计算负样本个数
        neg_roi_per_this_image = int(min(neg_roi_per_this_image,
                                         neg_index.size))
        if neg_index.size > 0:
            neg_index = np.random.choice(
                neg_index, size=neg_roi_per_this_image, replace=False)#（不放回抽样）随机选择设定负样本个数的负样本

        # The indices that we're selecting (both positive and negative).
        keep_index = np.append(pos_index, neg_index) #所有选取的样本index
        gt_roi_label = gt_roi_label[keep_index]#对应的label
        gt_roi_label[pos_roi_per_this_image:] = 0  # negative labels --> 0
        sample_roi = roi[keep_index] #根据keep_index选出对应roi框

        # Compute offsets and scales to match sampled RoIs to the GTs.
        gt_roi_loc = bbox2loc(sample_roi, bbox[gt_assignment[keep_index]]) #计算我们取样的roi和真实bbox 的loc 便于后面Loss的计算
        gt_roi_loc = ((gt_roi_loc - np.array(loc_normalize_mean, np.float32)
                       ) / np.array(loc_normalize_std, np.float32))#标准化

        return sample_roi, gt_roi_loc, gt_roi_label

#从20000多个Anchor中 选择正负样本128共256个进行训练RPVN  对anchor进行9*2(前，后景)的分类任务 9*4(ymax,xmax,ymin,xmin)个位置回归任务
#规定IOU最高为正 IOU>0.7为正样本，IOU<0.3为负样本 只计算前景损失
class AnchorTargetCreator(object):
    """Assign the ground truth bounding boxes to anchors.

    Assigns the ground truth bounding boxes to anchors for training Region
    Proposal Networks introduced in Faster R-CNN [#]_.

    Offsets and scales to match anchors to the ground truth are
    calculated using the encoding scheme of
    :func:`model.utils.bbox_tools.bbox2loc`.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        n_sample (int): The number of regions to produce.
        pos_iou_thresh (float): Anchors with IoU above this
            threshold will be assigned as positive.
        neg_iou_thresh (float): Anchors with IoU below this
            threshold will be assigned as negative.
        pos_ratio (float): Ratio of positive regions in the
            sampled regions.

    """

    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        """Assign ground truth supervision to sampled subset of anchors.

        Types of input arrays and output arrays are same.

        Here are notations.

        * :math:`S` is the number of anchors.
        * :math:`R` is the number of bounding boxes.

        Args:
            bbox (array): Coordinates of bounding boxes. Its shape is
                :math:`(R, 4)`.
            anchor (array): Coordinates of anchors. Its shape is
                :math:`(S, 4)`.
            img_size (tuple of ints): A tuple :obj:`H, W`, which
                is a tuple of height and width of an image.

        Returns:
            (array, array):

            #NOTE: it's scale not only  offset
            * **loc**: Offsets and scales to match the anchors to \
                the ground truth bounding boxes. Its shape is :math:`(S, 4)`.
            * **label**: Labels of anchors with values \
                :obj:`(1=positive, 0=negative, -1=ignore)`. Its shape \
                is :math:`(S,)`.

        """

        img_H, img_W = img_size  #图片高 宽

        n_anchor = len(anchor) #anchor个数
        inside_index = _get_inside_index(anchor, img_H, img_W) #将超出图片边界的anchor过滤 返回在界内的anchor索引
        anchor = anchor[inside_index]#选择界内的anchor
        argmax_ious, label = self._create_label(
            inside_index, anchor, bbox)#根据真实bbox  给anchor创建标签label 并返回标签对应的[(iou最大值的gt_box)的index]

        # compute bounding box regression targets
        loc = bbox2loc(anchor, bbox[argmax_ious]) #计算anchor和自己对应IOU最大的gt_bbox的偏差

        # map up to original set of anchors
        label = _unmap(label, n_anchor, inside_index, fill=-1)#将label数据 置于原来总anchor的位置，并将其余没选出的位置置为-1
        loc = _unmap(loc, n_anchor, inside_index, fill=0)  #将loc数据 置于原来总anchor的位置，并将其余没选出的位置置为0

        return loc, label

    def _create_label(self, inside_index, anchor, bbox):
        # label: 1 is positive, 0 is negative, -1 is dont care
        label = np.empty((len(inside_index),), dtype=np.int32) #len(边界内的anochor)
        label.fill(-1)

        argmax_ious, max_ious, gt_argmax_ious = \
            self._calc_ious(anchor, bbox, inside_index)

        # assign negative labels first so that positive labels can clobber them
        label[max_ious < self.neg_iou_thresh] = 0 #将IOU小于阀值的置为0，negative

        # positive label: for each gt, anchor with highest iou
        label[gt_argmax_ious] = 1  #将于gt_box IOU最大的anchor label置为1

        # positive label: above threshold IOU
        label[max_ious >= self.pos_iou_thresh] = 1 #IOU大于阀值的置为1

        # subsample positive labels if we have too many
        n_pos = int(self.pos_ratio * self.n_sample)  #我们要的正样本数
        pos_index = np.where(label == 1)[0] #实际正样本数,包括所有大于pos阈值和与gt交并比最大的框。
        if len(pos_index) > n_pos: #实际>我们要的
            disable_index = np.random.choice(
                pos_index, size=(len(pos_index) - n_pos), replace=False)  #从中随机选择出不要的样本
            label[disable_index] = -1 #将不要的样本置为-1  ，dont care

        # subsample negative labels if we have too many
        n_neg = self.n_sample - np.sum(label == 1) #我们要的负样本数
        neg_index = np.where(label == 0)[0]
        if len(neg_index) > n_neg:
            disable_index = np.random.choice(
                neg_index, size=(len(neg_index) - n_neg), replace=False)
            label[disable_index] = -1

        return argmax_ious, label

    def _calc_ious(self, anchor, bbox, inside_index):
        # ious between the anchors and the gt boxes
        ious = bbox_iou(anchor, bbox)  #计算IOU 
        argmax_ious = ious.argmax(axis=1) #argmax_ious[i]=j 表示第i个anchor与第j个bbox IOU值最大
        max_ious = ious[np.arange(len(inside_index)), argmax_ious] #max_ious[i]=j 表示第i个anchor与对应gt_box IOU的最大|值|是 j
        gt_argmax_ious = ious.argmax(axis=0) #gt_argmax_ious[i]=j 表示第i个bbox与第j个anchor IOU值最大
        gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])] #max_ious[i]=j 表示第i个gt_bbox与对应anchor IOU的最大值是 j
        gt_argmax_ious = np.where(ious == gt_max_ious)[0] #是一个索引gt_argmax_ious[a,b,c,d,e] 表示第a个anchor与对应box IOU值最大
                                                 #其中a<=b<=c<=d<=e即表示我们将要选出的框的索引 表示a这个anchor和某个bbox IOU最大
       #这已经够了，我们要选他，并不关心他与哪个框最大。所以为什么不取gt_argmax_ious = ious.argmax(axis=0) 的结果呢，
    #   因为他没有把等值算上，比如第二个box与 第20个anchor和第300个anchor IOU最大都是0.9 前面的运算只会选择第20个anchor 而不会选择300  
        return argmax_ious, max_ious, gt_argmax_ious


def _unmap(data, count, index, fill=0):
    # Unmap a subset of item (data) back to the original set of items (of
    # size count)

    if len(data.shape) == 1: #1维array
        ret = np.empty((count,), dtype=data.dtype)
        ret.fill(fill)#填满fill
        ret[index] = data #索引处换为数据
    else: #维度对齐
        ret = np.empty((count,) + data.shape[1:], dtype=data.dtype)#根据data的shape创建 np.array
        ret.fill(fill)
        ret[index, :] = data
    return ret


def _get_inside_index(anchor, H, W):   #返回anchor在图片内（不超出边界）的anchor索引
    # Calc indicies of anchors which are located completely inside of the image
    # whose size is speficied.
    index_inside = np.where(
        (anchor[:, 0] >= 0) &
        (anchor[:, 1] >= 0) &
        (anchor[:, 2] <= H) &
        (anchor[:, 3] <= W)
    )[0]
    return index_inside

#计算所有anchor是前景的概率，选择概率大的12,000（训练）/6,000（测试）个 ,修正位置参数，获得ROIS 再经过NMS后
#选择2,000/300 个为真正的ROIS 输出（2000/300,R）的ROIS
class ProposalCreator:
    # unNOTE: I'll make it undifferential
    # unTODO: make sure it's ok
    # It's ok
    """Proposal regions are generated by calling this object.

    The :meth:`__call__` of this object outputs object detection proposals by
    applying estimated bounding box offsets
    to a set of anchors.

    This class takes parameters to control number of bounding boxes to
    pass to NMS and keep after NMS.
    If the paramters are negative, it uses all the bounding boxes supplied
    or keep all the bounding boxes returned by NMS.

    This class is used for Region Proposal Networks introduced in
    Faster R-CNN [#]_.

    .. [#] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        nms_thresh (float): Threshold value used when calling NMS.
        n_train_pre_nms (int): Number of top scored bounding boxes
            to keep before passing to NMS in train mode.
        n_train_post_nms (int): Number of top scored bounding boxes
            to keep after passing to NMS in train mode.
        n_test_pre_nms (int): Number of top scored bounding boxes
            to keep before passing to NMS in test mode.
        n_test_post_nms (int): Number of top scored bounding boxes
            to keep after passing to NMS in test mode.
        force_cpu_nms (bool): If this is :obj:`True`,
            always use NMS in CPU mode. If :obj:`False`,
            the NMS mode is selected based on the type of inputs.
        min_size (int): A paramter to determine the threshold on
            discarding bounding boxes based on their sizes.

    """

    def __init__(self,
                 parent_model,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 min_size=16
                 ):
        self.parent_model = parent_model
        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.min_size = min_size

    def __call__(self, loc, score, #传入，预测的loc补偿，score分数，featuremap的所有anchor 
                 anchor, img_size, scale=1.):
        """input should  be ndarray
        Propose RoIs.

        Inputs :obj:`loc, score, anchor` refer to the same anchor when indexed
        by the same index.

        On notations, :math:`R` is the total number of anchors. This is equal
        to product of the height and the width of an image and the number of
        anchor bases per pixel.

        Type of the output is same as the inputs.

        Args:
            loc (array): Predicted offsets and scaling to anchors.
                Its shape is :math:`(R, 4)`.
            score (array): Predicted foreground probability for anchors.
                Its shape is :math:`(R,)`.
            anchor (array): Coordinates of anchors. Its shape is
                :math:`(R, 4)`.
            img_size (tuple of ints): A tuple :obj:`height, width`,
                which contains image size after scaling.
            scale (float): The scaling factor used to scale an image after
                reading it from a file.

        Returns:
            array:
            An array of coordinates of proposal boxes.
            Its shape is :math:`(S, 4)`. :math:`S` is less than
            :obj:`self.n_test_post_nms` in test time and less than
            :obj:`self.n_train_post_nms` in train time. :math:`S` depends on
            the size of the predicted bounding boxes and the number of
            bounding boxes discarded by NMS.

        """
        # NOTE: when test, remember
        # faster_rcnn.eval()
        # to set self.traing = False
        if self.parent_model.training:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # Convert anchors into proposal via bbox transformations.
        # roi = loc2bbox(anchor, loc)
        roi = loc2bbox(anchor, loc) #anchor调整位置后获得roi

        # Clip predicted boxes to image.限制(np.clip())调整后得到的roi候选框在图片内(防止越界）
        roi[:, slice(0, 4, 2)] = np.clip(
            roi[:, slice(0, 4, 2)], 0, img_size[0]) # 0<ymax,ymin<H  小于0部分变为0 大于H部分变为H
        roi[:, slice(1, 4, 2)] = np.clip(
            roi[:, slice(1, 4, 2)], 0, img_size[1]) # 0<xmax,xmin<W  小于0部分变为0 大于W部分变为W

        # Remove predicted boxes with either height or width < threshold.   threshold 框最小阀值，去除比较小的rois框
        min_size = self.min_size * scale
        hs = roi[:, 2] - roi[:, 0]
        ws = roi[:, 3] - roi[:, 1]
        keep = np.where((hs >= min_size) & (ws >= min_size))[0]
        roi = roi[keep, :]
        score = score[keep]

        # Sort all (proposal, score) pairs by score from highest to lowest.
        # Take top pre_nms_topN (e.g. 6000).
        order = score.ravel().argsort()[::-1]  #分数由高到低排序
        if n_pre_nms > 0:                   #进入nms前 我们想保留的ROIS个数
            order = order[:n_pre_nms]
        roi = roi[order, :]

        # Apply nms (e.g. threshold = 0.7).
        # Take after_nms_topN (e.g. 300).

        # unNOTE: somthing is wrong here!
        # TODO: remove cuda.to_gpu
        keep = non_maximum_suppression(
            cp.ascontiguousarray(cp.asarray(roi)),
            thresh=self.nms_thresh)
        if n_post_nms > 0:  #我们想保留的nms后ROIS的个数
            keep = keep[:n_post_nms]
        roi = roi[keep]
        return roi
