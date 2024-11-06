import os, sys
sys.path.append(os.getcwd())
from typing import List, Tuple

import numpy as np

from dsnet.src.helpers import bbox_helper


def get_anchors(picks, picks_shape, scales: List[int]) -> np.ndarray:
    """Generate all multi-scale anchors for a sequence in center-width format.

    :param seq_len: Sequence length.
    :param scales: List of bounding box widths.
    :return: All anchors in center-width format.
    """
    anchors = np.zeros((picks_shape[0], picks_shape[1], len(scales), 2), dtype=np.int32)
    # import ipdb; ipdb.set_trace()
    picks = picks.detach().cpu().numpy()
    for pos_0 in range(picks_shape[0]):
        for pos_1 in range(picks_shape[1]):   
            for scale_idx, scale in enumerate(scales):
                anchors[pos_0][pos_1][scale_idx] = [picks[pos_0][pos_1], scale]
    # import ipdb; ipdb.set_trace()
    return anchors

def preprocess_anchors(anchor, max_length):
    """Preprocess anchors so that they are within the range of the sequence.

    :param anchor: anchor in center-width format
    """
    # import ipdb; ipdb.set_trace()
    anchor_shape = anchor.shape
    preprocess_anchors = bbox_helper.cw2lr(anchor)
    expanded_max_length = np.array(max_length)[:, None, None, None]
    max_length = np.tile(expanded_max_length, (1,anchor_shape[1],anchor_shape[2],anchor_shape[3]))
    max_length = max_length.reshape((-1,2))
    indices_left = np.where(preprocess_anchors < 0)
    indices_right = np.where(preprocess_anchors > max_length)
    preprocess_anchors[indices_left] = 0
    # indices_right = (np.array([0,1]), np.array([0,0]))
    preprocess_anchors[indices_right] = max_length[indices_right]
    # if not indices_right[0].size == 0:
        # import ipdb; ipdb.set_trace()
        # preprocess_anchors[indices_right] = max_length[indices_right] # only revise max_length
        # preprocess_anchors[indices_right[0]] = 0 # only revise max_length
    # import ipdb; ipdb.set_trace()
    preprocess_anchors = np.array(bbox_helper.lr2cw(preprocess_anchors))
    preprocess_anchors = preprocess_anchors.reshape(anchor_shape)
    
    return preprocess_anchors
    

def get_adapted_pos(iou, iou_thresh):
    # import ipdb; ipdb.set_trace()
    # 阈值 应该基于每个anchor和target的iou来确定，把所有的分成正样本和负样本 正：负 1：2
    # import ipdb; ipdb.set_trace()
    all_idx, = np.where(iou > 0)
    all_idx_sort = np.argsort(iou[all_idx])[::-1] # 从大到小
    # pos_num = int(len(all_idx)*(1-iou_thresh))
    pos_num = 16
    pos_idx = all_idx_sort[:pos_num]
    return pos_idx
    
def get_adapted_incomplete(iou,iou_thresh):
    if iou_thresh == 0:
        return np.where(iou > 0)
    else:
        all_idx, = np.where(iou > 0)
        all_idx_sort = np.argsort(iou[all_idx]) # 从小到大
        incomplete_num = int(len(all_idx) * iou_thresh)
        incomplete_idx = all_idx_sort[:incomplete_num]
        return incomplete_idx
def get_pos_label(anchors: np.ndarray,
                  targets: np.ndarray,
                  iou_thresh: float,
                  adapted_sample:dict=None) -> Tuple[np.ndarray, np.ndarray]:
                  
    """Generate positive samples for training.

    :param anchors: List of CW anchors
    :param targets: List of CW target bounding boxes
    :param iou_thresh: If IoU between a target bounding box and any anchor is
        higher than this threshold, the target is regarded as a positive sample.
    :return: Class and location offset labels
    """
    bs, seq_len, num_scales, _ = anchors.shape
    anchors = np.reshape(anchors, (bs, seq_len * num_scales, 2)) # (32, 375, 2)
    loc_label = np.zeros((bs, seq_len * num_scales, 2)) # (32, 375, 2)
    cls_label = np.zeros((bs, seq_len * num_scales), dtype=np.int32) # (32, 375)
    adapted_sample_dict = {}
    for i in range(bs):
        target_list = []
        for target_data in targets[i]: 
            target = np.tile(target_data, (seq_len * num_scales, 1)) # (375, 2)
            # import ipdb;ipdb.set_trace()
            iou = bbox_helper.iou_cw(anchors[i], target)
            pos_idx, = np.where(iou > iou_thresh)
            if len(pos_idx) == 0 and adapted_sample is None:
                # import ipdb; ipdb.set_trace()
                target_list.append(target)
                # pos_idx, = np.where(iou > 0)
                pos_idx = get_adapted_pos(iou,iou_thresh)
            elif adapted_sample is not None:
                if i in adapted_sample.keys():
                    # incomplete_idx = get_adapted_incomplete(iou,iou_thresh)
                    pos_idx = get_adapted_incomplete(iou,iou_thresh)
            if len(target_list) !=0 :
                # import ipdb; ipdb.set_trace()
                adapted_sample_dict[i] = target_data
                # pos_idx_sort = np.argsort(iou[pos_idx]
            cls_label[i][pos_idx] = 1
            try:
                # loc_label[i][pos_idx] = bbox2offset(target[pos_idx], anchors[i][pos_idx])
                loc_label[i][pos_idx]  = bbox2offset(target[pos_idx], anchors[i][pos_idx])
            except IndexError:
                loc_label[i][pos_idx] = np.zeros((0,2))
    loc_label = loc_label.reshape((bs, seq_len, num_scales, 2))
    cls_label = cls_label.reshape((bs, seq_len, num_scales))
    # import ipdb; ipdb.set_trace()
    return cls_label, loc_label, adapted_sample_dict


def get_neg_label(cls_label: np.ndarray, num_neg) -> np.ndarray:
    """Generate random negative samples.

    :param cls_label: Class labels including only positive samples.
    :param num_neg: Number of negative samples.
    :return: Label with original positive samples (marked by 1), negative
        samples (marked by -1), and ignored samples (marked by 0)
    """
    # import ipdb; ipdb.set_trace()
    bs, seq_len, num_scales = cls_label.shape # (32, 75, 5)
    cls_label = cls_label.copy().reshape(bs, -1)
    cls_label[:][cls_label < 0] = 0  # reset negative samples
    # import ipdb; ipdb.set_trace()
    for i in range(bs):
        neg_idx, = np.where(cls_label[i] == 0)
        np.random.shuffle(neg_idx)
        neg_idx = neg_idx[:num_neg[i]]
        cls_label[i][neg_idx] = -1
    cls_label = np.reshape(cls_label, (bs, seq_len, num_scales))
 
    
    return cls_label


def offset2bbox(offsets: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Convert predicted offsets to CW bounding boxes.

    :param offsets: Predicted offsets.
    :param anchors: Sequence anchors.
    :return: Predicted bounding boxes.
    """
    offsets = offsets.reshape(len(offsets),-1, 2)
    anchors = anchors.reshape(len(anchors),-1, 2)

    offset_center, offset_width = offsets[:, :, 0], offsets[:, :, 1]
    anchor_center, anchor_width = anchors[:, :, 0], anchors[:, :, 1]

    # Tc = Oc * Aw + Ac
    bbox_center = offset_center * anchor_width + anchor_center
    # Tw = exp(Ow) * Aw
    bbox_width = np.exp(offset_width) * anchor_width

    bbox = np.stack((bbox_center, bbox_width),axis = 2)
    return bbox


def bbox2offset(bboxes: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Convert bounding boxes to offset labels.

    :param bboxes: List of CW bounding boxes.
    :param anchors: List of CW anchors.
    :return: Offsets labels for training.
    """
    bbox_center, bbox_width = bboxes[:, 0], bboxes[:, 1]
    anchor_center, anchor_width = anchors[:, 0], anchors[:, 1]
    # anchor_width = anchor_width
    # Oc = (Tc - Ac) / Aw
    offset_center = (bbox_center - anchor_center) / anchor_width
    # Ow = ln(Tw / Aw)
    offset_width = np.log(bbox_width / anchor_width)

    offset = np.vstack((offset_center, offset_width)).T
    return offset
