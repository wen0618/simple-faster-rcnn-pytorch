from __future__ import division

from collections import defaultdict
import itertools
import numpy as np
import six

from model.utils.bbox_tools import bbox_iou

#根据PASCAL VOC的evaluation code 计算平均精度
def eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
        gt_difficults=None,
        iou_thresh=0.5, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.

    This function evaluates predicted bounding boxes obtained from a dataset
    which has :math:`N` images by using average precision for each class.
    The code is based on the evaluation code used in PASCAL VOC Challenge.
#test_num张图片（图片数据来自测试数据testdata）的预测框，标签，分数，和真实的框，标签和分数。
#        所有参数都是list len（list）=opt.test_num(default=10000)
#        pred_boxes:[(A,4),(B,4),(C,4)....共test_num个] 输入源gt_数据 经过train.predict函数预测出的结果框
#        pred_labels[(A,),(B,),(C,)...共test_num个]  pred_scores同pred_labels  
#        A,B,C,D是由nms决定的个数，即预测的框个数，不确定。
#        gt_bboxes：[(a,4),(b,4)....共test_num个]  a b...是每张图片标注真实框的个数
#        gt_labels与gt_difficults同理
#        use_07_metric (bool): 是否使用PASCAL VOC 2007 evaluation metric计算平均精度

    Args:
        pred_bboxes (iterable of numpy.ndarray): An iterable of :math:`N`
            sets of bounding boxes.
            Its index corresponds to an index for the base dataset.
            Each element of :obj:`pred_bboxes` is a set of coordinates
            of bounding boxes. This is an array whose shape is :math:`(R, 4)`,
            where :math:`R` corresponds
            to the number of bounding boxes, which may vary among boxes.
            The second axis corresponds to
            :math:`y_{min}, x_{min}, y_{max}, x_{max}` of a bounding box.
        pred_labels (iterable of numpy.ndarray): An iterable of labels.
            Similar to :obj:`pred_bboxes`, its index corresponds to an
            index for the base dataset. Its length is :math:`N`.
        pred_scores (iterable of numpy.ndarray): An iterable of confidence
            scores for predicted bounding boxes. Similar to :obj:`pred_bboxes`,
            its index corresponds to an index for the base dataset.
            Its length is :math:`N`.
        gt_bboxes (iterable of numpy.ndarray): An iterable of ground truth
            bounding boxes
            whose length is :math:`N`. An element of :obj:`gt_bboxes` is a
            bounding box whose shape is :math:`(R, 4)`. Note that the number of
            bounding boxes in each image does not need to be same as the number
            of corresponding predicted boxes.
        gt_labels (iterable of numpy.ndarray): An iterable of ground truth
            labels which are organized similarly to :obj:`gt_bboxes`.
        gt_difficults (iterable of numpy.ndarray): An iterable of boolean
            arrays which is organized similarly to :obj:`gt_bboxes`.
            This tells whether the
            corresponding ground truth bounding box is difficult or not.
            By default, this is :obj:`None`. In that case, this function
            considers all bounding boxes to be not difficult.
        iou_thresh (float): A prediction is correct if its Intersection over
            Union with the ground truth is above this value.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.

    Returns:
        dict:

        The keys, value-types and the description of the values are listed
        below.

        * **ap** (*numpy.ndarray*): An array of average precisions. \
            The :math:`l`-th value corresponds to the average precision \
            for class :math:`l`. If class :math:`l` does not exist in \
            either :obj:`pred_labels` or :obj:`gt_labels`, the corresponding \
            value is set to :obj:`numpy.nan`.
        * **map** (*float*): The average of Average Precisions over classes.

    """

    prec, rec = calc_detection_voc_prec_rec(     #函数算出每个label类的准确率和召回率,定义见下
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        iou_thresh=iou_thresh)

    ap = calc_detection_voc_ap(prec, rec, use_07_metric=use_07_metric)  #根据prec和rec 计算ap和map

    return {'ap': ap, 'map': np.nanmean(ap)}


def calc_detection_voc_prec_rec(
        pred_bboxes, pred_labels, pred_scores, gt_bboxes, gt_labels,
        gt_difficults=None,
        iou_thresh=0.5):
    """Calculate precision and recall based on evaluation code of PASCAL VOC.

    This function calculates precision and recall of
    predicted bounding boxes obtained from a dataset which has :math:`N`
    images.
    The code is based on the evaluation code used in PASCAL VOC Challenge.

    Args:
        pred_bboxes (iterable of numpy.ndarray): An iterable of :math:`N`
            sets of bounding boxes.
            Its index corresponds to an index for the base dataset.
            Each element of :obj:`pred_bboxes` is a set of coordinates
            of bounding boxes. This is an array whose shape is :math:`(R, 4)`,
            where :math:`R` corresponds
            to the number of bounding boxes, which may vary among boxes.
            The second axis corresponds to
            :math:`y_{min}, x_{min}, y_{max}, x_{max}` of a bounding box.
        pred_labels (iterable of numpy.ndarray): An iterable of labels.
            Similar to :obj:`pred_bboxes`, its index corresponds to an
            index for the base dataset. Its length is :math:`N`.
        pred_scores (iterable of numpy.ndarray): An iterable of confidence
            scores for predicted bounding boxes. Similar to :obj:`pred_bboxes`,
            its index corresponds to an index for the base dataset.
            Its length is :math:`N`.
        gt_bboxes (iterable of numpy.ndarray): An iterable of ground truth
            bounding boxes
            whose length is :math:`N`. An element of :obj:`gt_bboxes` is a
            bounding box whose shape is :math:`(R, 4)`. Note that the number of
            bounding boxes in each image does not need to be same as the number
            of corresponding predicted boxes.
        gt_labels (iterable of numpy.ndarray): An iterable of ground truth
            labels which are organized similarly to :obj:`gt_bboxes`.
        gt_difficults (iterable of numpy.ndarray): An iterable of boolean
            arrays which is organized similarly to :obj:`gt_bboxes`.
            This tells whether the
            corresponding ground truth bounding box is difficult or not.
            By default, this is :obj:`None`. In that case, this function
            considers all bounding boxes to be not difficult.
        iou_thresh (float): A prediction is correct if its Intersection over
            Union with the ground truth is above this value..

    Returns:
        tuple of two lists:
        This function returns two lists: :obj:`prec` and :obj:`rec`.

        * :obj:`prec`: A list of arrays. :obj:`prec[l]` is precision \
            for class :math:`l`. If class :math:`l` does not exist in \
            either :obj:`pred_labels` or :obj:`gt_labels`, :obj:`prec[l]` is \
            set to :obj:`None`.
        * :obj:`rec`: A list of arrays. :obj:`rec[l]` is recall \
            for class :math:`l`. If class :math:`l` that is not marked as \
            difficult does not exist in \
            :obj:`gt_labels`, :obj:`rec[l]` is \
            set to :obj:`None`.

    """

    pred_bboxes = iter(pred_bboxes) #iter()生成迭代器 eg.lst = [1, 2, 3]  for i in iter(lst):  print(i) 1 2 3
    pred_labels = iter(pred_labels)
    pred_scores = iter(pred_scores)
    gt_bboxes = iter(gt_bboxes)
    gt_labels = iter(gt_labels)
    if gt_difficults is None:
        gt_difficults = itertools.repeat(None) #/itertools.repeat生成一个重复的迭代器 None是每次迭代获得的数据
    else:
        gt_difficults = iter(gt_difficults)

    n_pos = defaultdict(int) #defaultdict():defaultdict的作用是在于，当字典里的key不存在但被查找时，返回的不是keyError而是一个默认值，
                                     #dict[key]=default(int)=0 default(list)=[] default(dict)={}
    score = defaultdict(list)
    match = defaultdict(list)

    for pred_bbox, pred_label, pred_score, gt_bbox, gt_label, gt_difficult in \
            six.moves.zip(             
                pred_bboxes, pred_labels, pred_scores,
                gt_bboxes, gt_labels, gt_difficults):
#six.move.zip()==zip():函数用于将可迭代的对象作为参数，将对象中||对应的||元素打包成一个个元组，然后返回由这些元组组成的列表。 *zip()解压


        if gt_difficult is None:
            gt_difficult = np.zeros(gt_bbox.shape[0], dtype=bool)#全部设置为非difficult
        
#遍历一张图片中 所有出现的label:
        for l in np.unique(np.concatenate((pred_label, gt_label)).astype(int)): 
       #拼接后返回无重复的从小到大排序的一维numpy ,如[2,3,4,5,6] 并遍历这个一维数组，即遍历这张图片出现过的标签数字(gt_label+pred_label)
       #对于其中的每一个标签l：
        
            pred_mask_l = pred_label == l #广播(对矩阵中每个元素执行相同的操作)
                                          #pred_mask_l=[eg. T,F,T,T,F,F,F,T..] 所有预测label中等于L的为T 否则F
            pred_bbox_l = pred_bbox[pred_mask_l] #选出label=L的所有pre_box
            pred_score_l = pred_score[pred_mask_l]#label=L 对应所有pre_score
            # sort by score
            order = pred_score_l.argsort()[::-1] #获得score降序排序 索引||argsort()返回的是索引
            pred_bbox_l = pred_bbox_l[order] #按照分数从高到低 对box进行排序
            pred_score_l = pred_score_l[order]#对score进行排序

            gt_mask_l = gt_label == l #广播gt_mask_l =[eg. T,F,T,T,F,F,F,T..]  所有真实label中等于L的为T 否则F
            gt_bbox_l = gt_bbox[gt_mask_l]#选出label=L的所有gt_box
            gt_difficult_l = gt_difficult[gt_mask_l]#选出label=L的所有difficult

            n_pos[l] += np.logical_not(gt_difficult_l).sum()#对T,F取反求和 即统计difficult=0的个数
            score[l].extend(pred_score_l) #score={l:predscore_1,....}  extend是针对list的方法

            if len(pred_bbox_l) == 0: #没有预测的label=L的box 即真实label有L，我们全没有预测到 
                continue              #跳过这张图片 此时没有对match字典操作 之前score[l].extend操作也为空 保持了match和score形状一致
            if len(gt_bbox_l) == 0:   #没有真实的label=L的box 即预测label有L，真实中没有 我们都预测错了
                match[l].extend((0,)  #pred_bbox_l.shape[0]) #match{L:[0，0，0，..  n_pred_box个0]}
                continue              #预测错label就是0 已不需要后续操作 跳过此图片

            # VOC evaluation follows integer typed bounding boxes.
            # 我不太懂这么做的目的，作者给的注释是follows integer typed bounding boxes
             # 但是只改变了ymax,xmax的值，重要的是这样做并不能转化为整数 pred_bbox和gt_bbox只
             #   参与了IOU计算且后面没有参与其他计算 有待解答。
               
            pred_bbox_l = pred_bbox_l.copy()
            pred_bbox_l[:, 2:] += 1  #ymax,xmax +=1
            gt_bbox_l = gt_bbox_l.copy()
            gt_bbox_l[:, 2:] += 1 #ymax,xmax +=1

            iou = bbox_iou(pred_bbox_l, gt_bbox_l) #计算两个box的IOU
            gt_index = iou.argmax(axis=1) #argmax(axis=1)按照a[0][1]中的a[1]方向，即行方向搜索最大值
                                #有len(pred_bbox_l)个索引 第i个索引值n表示 gt_box[n]与pred_box[i] IOU最大？
            # set -1 if there is no matching ground truth
            gt_index[iou.max(axis=1) < iou_thresh] = -1 #将gt_box与pred_box iou<thresh的索引值置为-1
                                                         #即针对每个pred_bbox，与每个gt_bbox IOU的最大值 如果最大值小于阀值则置为-1
                                                         #即我们预测的这个box效果并不理想 后续会将index=-1的 matchlabel=0

            del iou

            selec = np.zeros(gt_bbox_l.shape[0], dtype=bool)
            for gt_idx in gt_index: #遍历gt_index索引值
                if gt_idx >= 0: #即IOU满足条件
                    if gt_difficult_l[gt_idx]: #对应的gt_difficult =1 即困难标识
                        match[l].append(-1)     #match[l]追加一个-1
                    else:                  #不是困难标识
                        if not selec[gt_idx]:   #没有被选过 select[gt_idx]=0时
                            match[l].append(1) #match[l]追加一个1
                        else:              #对应的gt_box已经被选择过一次 即已经和前面某pred_box IOU最大
                            match[l].append(0) #match[l]追加一个0
                    selec[gt_idx] = True    #select[gt_idx]=1  置为1，表示已经被选过一次
                else:       #不满足IOU>thresh 效果并不理想
                    match[l].append(0) #match[l]追加一个0
#我们注意到 上面为每个pred_box都打了label 0,1,-1  len(match[l])=len(score[l])=len(pred_bbox_l)
                                
    for iter_ in (  #上面的 six.moves.zip遍历会在某一iter遍历到头后停止，由于pred_bboxes等是全局iter对象
            #我们此时继续调用next取下一数据，如果有任一数据不为None，那么说明他们的len是不相等的 有悖常理，数据错误
            pred_bboxes, pred_labels, pred_scores,
            gt_bboxes, gt_labels, gt_difficults):
        if next(iter_, None) is not None: #next(iter,None) 表示调用next 如果已经遍历到头 不抛出异常而是返回None
            raise ValueError('Length of input iterables need to be same.')
#注意pr曲线与roc曲线的区别
    n_fg_class = max(n_pos.keys()) + 1 #有n_fg_class个类
    prec = [None] * n_fg_class       #list[None,.....len(n_fg_class)]
    rec = [None] * n_fg_class

    for l in n_pos.keys():          #遍历所有Label
        score_l = np.array(score[l]) #list to np.array
        match_l = np.array(match[l], dtype=np.int8)

        order = score_l.argsort()[::-1]
        match_l = match_l[order]  #对应match按照 score由大到小排序

        tp = np.cumsum(match_l == 1) #统计累计 match_1=1的个数
        fp = np.cumsum(match_l == 0) #统计累计 match_1=0的个数
#tp eg. [1 2 3 3 4 5 5 6 7 8 8 9 10 11........]
#fp eg. [0 0 0 0 0 0 0 1 1 1 1 1 1 2 ......]
# 如果 fp + tp = 0, 那么计算出的prec[l] = nan

        # If an element of fp + tp is 0,
        # the corresponding element of prec[l] is nan.
        prec[l] = tp / (fp + tp)  //计算准确率
        # If n_pos[l] is 0, rec[l] is None.
        if n_pos[l] > 0:  #w/如果n_pos[l] = 0,那么rec[l] =Non
            rec[l] = tp / n_pos[l] #/计算召回率

    return prec, rec


def calc_detection_voc_ap(prec, rec, use_07_metric=False):
    """Calculate average precisions based on evaluation code of PASCAL VOC.

    This function calculates average precisions
    from given precisions and recalls.
    The code is based on the evaluation code used in PASCAL VOC Challenge.

    Args:
        prec (list of numpy.array): A list of arrays.
            :obj:`prec[l]` indicates precision for class :math:`l`.
            If :obj:`prec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        rec (list of numpy.array): A list of arrays.
            :obj:`rec[l]` indicates recall for class :math:`l`.
            If :obj:`rec[l]` is :obj:`None`, this function returns
            :obj:`numpy.nan` for class :math:`l`.
        use_07_metric (bool): Whether to use PASCAL VOC 2007 evaluation metric
            for calculating average precision. The default value is
            :obj:`False`.

    Returns:
        ~numpy.ndarray:
        This function returns an array of average precisions.
        The :math:`l`-th value corresponds to the average precision
        for class :math:`l`. If :obj:`prec[l]` or :obj:`rec[l]` is
        :obj:`None`, the corresponding value is set to :obj:`numpy.nan`.

    """

    n_fg_class = len(prec)
    ap = np.empty(n_fg_class)
    for l in six.moves.range(n_fg_class):  #遍历每个label
        if prec[l] is None or rec[l] is None: #如果为None 则ap置为np.nan
            ap[l] = np.nan
            continue

        if use_07_metric:
            # 11 point metric
            ap[l] = 0
            for t in np.arange(0., 1.1, 0.1):  # //t=0 0.1 0.2 ....1.0
                if np.sum(rec[l] >= t) == 0:#这个标签的召回率没有>=t的
                    p = 0
                else:
                    p = np.max(np.nan_to_num(prec[l])[rec[l] >= t])#p=(rec>=t时 对应index：prec中的最大值)
                                # P=（X>t时，X对应的Y:Y的最大值）   np.nan_to_num 是为了让None=0 以便计算
                ap[l] += p / 11
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mpre = np.concatenate(([0], np.nan_to_num(prec[l]), [0]))  #头尾填0
            mrec = np.concatenate(([0], rec[l], [1]))  #头填0 尾填1

            mpre = np.maximum.accumulate(mpre[::-1])[::-1]#我们知道 我们是按score由高到低排序的 而且我们给box打了label
                                                          #   0,1,-1   score高时1的概率会大，所以pre是累计降序的
                                                          #  而rec是累积升序的，那么此时将pre倒序再maxuim.ac
                                                          #   获得累积最大值，再倒序后 从小到大排序的累积最大值

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0] #差位比较，看哪里改变了recall的值 记录Index (x轴)

            # and sum (\Delta recall) * prec
            ap[l] = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]) #差值*mpre_max值 （x轴差值*y_max）

    return ap
