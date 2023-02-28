#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   iou_matching.py
@Time    :   2023/02/26 21:09:08
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   None
'''


import numpy as np

import linear_assignment


def iou(bbox, candidates):
    """
    计算交并比IoU
    
    参数
    ----------
    bbox : ndarray
        一个边界框, 格式为 `(top left x, top left y, width, height)`.
    candidates : ndarray
        一个候选边界框矩阵 (每一行) 与 `bbox`相同的格式

    返回
    -------
    ndarray
        `bbox`和每一个候选框的IoU在 [0, 1] 之间。分数越高, 意味着 `bbox` 被候选框遮挡的分数越大。
    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]                                # 跟踪对象的左上角和右下角坐标
    candidates_tl = candidates[:, :2]                                               # 所有匹配候选检测框的左上角坐标
    candidates_br = candidates[:, :2] + candidates[:, 2:]                           # 所有匹配候选检测框的右下角坐标
    
    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]          # 交集的左上角坐标
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]          # 交集的右下角坐标
    
    wh = np.maximum(0., br - tl)                                                    # 交集的宽高
    
    area_intersection = wh.prod(axis=1)                                             # 交集的面积
    area_bbox = bbox[2:].prod()                                                     # Bbox的面积
    area_candidates = candidates[:, 2:].prod(axis=1)                                # 候选宽的面积
    
    return area_intersection / (area_bbox + area_candidates - area_intersection)    # IoU 
    





def iou_cost(tracks, detections, track_indices=None, detection_indices=None):
    """
    一个IoU距离度量。

    参数
    ----------
    tracks : List[deep_sort.track.Track]
        全部跟踪对象的列表
    detections : List[deep_sort.detection.Detection]
        当前检测对象的列表
    track_indices : Optional[List[int]]
        未匹配的跟踪对象的索引列表。默认是所有的 `tracks`.
    detection_indices : Optional[List[int]]
        未匹配的检测目标的索引。默认是所有的 `detections`.

    返回
    -------
    ndarray
        返回一个代价矩阵, 形状是 len(track_indices), len(detection_indices),其中 (i, j)处的元素是 
        `1 - iou(tracks[track_indices[i]], detections[detection_indices[j]])`。
    """
    
    if track_indices is None:
        track_indices = np.arange(len(tracks))
    if detection_indices is None:
        detection_indices = np.arange(len(detections))
    
    # 初始化代价矩阵
    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):                                      # 遍历每一个跟踪对象
        
        # 这里是为了加强保险, 能传递进来的都应该是time_since_update小于等于1的
        # 如果是传递进来的未匹配上的确认对象, time_since_update == 1
        # 如果是传递进来的未确认对象, time_since_update应该等于1, 不可能大于1
        # 因为如果第一次预测之后,未匹配上目标就会被设置为删除状态
        if tracks[track_idx].time_since_update > 1:                                      # 这里应该不会被执行                                   
            cost_matrix[row, :] = linear_assignment.INFTY_COST                           # 如果真的进来了, 表示不符合要求,不应该匹配他,设置个很大的代价
            continue

        bbox = tracks[track_idx].to_tlwh()                                               # 将当前的跟踪对象转换为tlwh  
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])         # 将检测目标坐标进行转换
        
        cost_matrix[row, :] = 1.0 - iou(bbox, candidates)                                # 填充代价矩阵
        
    return cost_matrix


















































