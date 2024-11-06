from itertools import groupby
from operator import itemgetter
from typing import Tuple

import numpy as np


def lr2cw(bbox_lr_list):
    """Convert bounding boxes from left-right (LR) to center-width (CW) format.

    :param bbox_lr: LR bounding boxes. Sized [N, 2].
    :return: CW bounding boxes. Sized [N, 2].
    """
    bbox_cw_list = []   
    for i in range(len(bbox_lr_list)):
        bbox_lr = np.asarray(bbox_lr_list[i], dtype=np.float32).reshape((-1, 2))
        center = (bbox_lr[:, 0] + bbox_lr[:, 1]) / 2
        width = bbox_lr[:, 1] - bbox_lr[:, 0]
        bbox_cw = np.vstack((center, width)).T
        bbox_cw_list.append(bbox_cw)
    return bbox_cw_list


def cw2lr(bbox_cw: np.ndarray) -> np.ndarray:
    """Convert bounding boxes from center-width (CW) to left-right (LR) format.

    :param bbox_cw: CW bounding boxes. Sized [N, 2].
    :return: LR bounding boxes. Sized [N, 2].
    """
    # bbox_cw = np.asarray(bbox_cw, dtype=np.float32).reshape((len(bbox_cw), -1, 2))
    bbox_cw = np.asarray(bbox_cw, dtype=np.float32).reshape((-1, 2))
    # left = bbox_cw[ :, :, 0] - bbox_cw[ :, :, 1] / 2
    # right = bbox_cw[ :, :, 0] + bbox_cw[ :, :, 1] / 2
    left = bbox_cw[ :, 0] - bbox_cw[ :, 1] / 2
    right = bbox_cw[ :, 0] + bbox_cw[ :, 1] / 2
    bbox_lr = np.vstack((left, right)).T
    return bbox_lr


def seq2bbox(sequence: np.ndarray) -> np.ndarray:
    """Generate CW bbox from binary sequence mask"""
    sequence = np.asarray(sequence, dtype=np.bool_)
    selected_indices = [] 
    for i in range(sequence.shape[0]):
        bool_seq, = np.where(sequence[i]==1)
        selected_indices.append(bool_seq)

    bboxes_lr_all = []
    for i in range(len(selected_indices)):
        bboxes_lr = []
        # for index,value in enumerate(selected_indices[i]):
        #     print(f"index: {index}, value: {value}")
        for k, g in groupby(enumerate(selected_indices[i]), lambda x: x[0] - x[1]):
            
            segment = list(map(itemgetter(1), g))
            start_frame, end_frame = segment[0], segment[-1] + 1
            bboxes_lr.append([start_frame, end_frame])
        bboxes_lr = np.asarray(bboxes_lr, dtype=np.int32)
        bboxes_lr_all.append(bboxes_lr) 
    return bboxes_lr_all


def iou_lr(anchor_bbox: np.ndarray, target_bbox: np.ndarray) -> np.ndarray:
    """Compute iou between multiple LR bbox pairs.

    :param anchor_bbox: LR anchor bboxes. Sized [N, 2].
    :param target_bbox: LR target bboxes. Sized [N, 2].
    :return: IoU between each bbox pair. Sized [N].
    """
    # max_length = 150
    # import ipdb; ipdb.set_trace()
    anchor_left, anchor_right = anchor_bbox[:, 0], anchor_bbox[:, 1]
    target_left, target_right = target_bbox[:, 0], target_bbox[:, 1]

    inter_left = np.maximum(anchor_left, target_left)
    inter_right = np.minimum(anchor_right, target_right)
    union_left = np.minimum(anchor_left, target_left)
    union_right = np.maximum(anchor_right, target_right)
    
    # inter_right[inter_right > max_length] = max_length 
    # union_right[union_right > max_length] = max_length 
        
    intersect = np.maximum(0, inter_right - inter_left)
    union = union_right - union_left
    union[union <= 0] = 1e-6

    iou = intersect / union
    return iou


def iou_cw(anchor_bbox: np.ndarray, target_bbox: np.ndarray) -> np.ndarray:
    """Compute iou between multiple CW bbox pairs. See ``iou_lr``"""
    anchor_bbox_lr = cw2lr(anchor_bbox)
    target_bbox_lr = cw2lr(target_bbox)
    # import ipdb; ipdb.set_trace()
    return iou_lr(anchor_bbox_lr, target_bbox_lr)


def nms(scores: np.ndarray,
        bboxes: np.ndarray,
        thresh: float) -> Tuple[np.ndarray, np.ndarray]:
    """Non-Maximum Suppression (NMS) algorithm on 1D bbox.

    :param scores: List of confidence scores for bboxes. Sized [N].
    :param bboxes: List of LR bboxes. Sized [N, 2].
    :param thresh: IoU threshold. Overlapped bboxes with IoU higher than this
        threshold will be filtered.
    :return: Remaining bboxes and its scores.
    """
    valid_idx = bboxes[:, 0] < bboxes[:, 1]
    scores = scores[valid_idx]
    bboxes = bboxes[valid_idx]

    arg_desc = scores.argsort()[::-1]

    scores_remain = scores[arg_desc]
    bboxes_remain = bboxes[arg_desc]

    keep_bboxes = []
    keep_scores = []

    while bboxes_remain.size > 0:
        bbox = bboxes_remain[0]
        score = scores_remain[0]
        keep_bboxes.append(bbox)
        keep_scores.append(score)

        iou = iou_lr(bboxes_remain, np.expand_dims(bbox, axis=0))

        keep_indices = (iou < thresh)
        bboxes_remain = bboxes_remain[keep_indices]
        scores_remain = scores_remain[keep_indices]
    # import ipdb; ipdb.set_trace()
    keep_bboxes = np.asarray(keep_bboxes, dtype=bboxes.dtype)
    keep_scores = np.asarray(keep_scores, dtype=scores.dtype)

    return keep_scores, keep_bboxes
