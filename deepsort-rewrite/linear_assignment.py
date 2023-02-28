#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   linear_assignment.py
@Time    :   2023/02/24 10:13:06
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   None
'''

from __future__ import absolute_import

import numpy as np

from scipy.optimize import linear_sum_assignment
import kalman_filter

INFTY_COST = 1e+5

# 最小代价匹配
def min_cost_matching(distance_metric, max_distance, tracks, detections, track_indices=None, detection_indices=None):
    
    if track_indices == None:
        track_indices = np.arange(len(tracks))
    if detection_indices == None:
        detection_indices = np.arange(len(detections))
    
    if len(track_indices) == 0 or len(detection_indices) == 0:                     # 当跟踪对象或者检测目标为空时, 将不会匹配任何目标
        return [], track_indices, detection_indices                                # 没有匹配上任何对象

    cost_matrix = distance_metric(tracks, detections, track_indices, detection_indices)         # 采用不同方法计算代价矩阵, 外观特征+马氏距离
    cost_matrix[cost_matrix > max_distance] = max_distance + 1e-5                               # 截断到最大值
    rows, cols = linear_sum_assignment(cost_matrix)                                             # 匈牙利匹配
    
    matches, unmatched_tracks, unmatched_detections = [], [], []
    for i, track_idx in enumerate(track_indices):                                               # 遍历需要匹配的跟踪对象                                           
        if i not in rows:                                                                       # 如果不在匹配列表里
            unmatched_tracks.append(track_idx)                                                  # 将未匹配跟踪对象的索引加入列表
    for j, detection_idx in enumerate(detection_indices):                                       # 遍历需要匹配的检测对象
        if j not in cols:                                                                       # 如果不在匹配列表
            unmatched_detections.append(detection_idx)                                          # 将未匹配的检测对象索引加入列表

    for row, col in zip(rows, cols):                                                            # 遍历所有匹配上的对象
        track_idx = track_indices[row]                                                          # 当前匹配的跟踪对象
        detection_idx = detection_indices[col]                                                  # 当前匹配的检测对象
        if cost_matrix[row, col] > max_distance:                                                # 如果代价矩阵的值超过阈值
            unmatched_tracks.append(track_idx)                                                  # 表示当前跟踪未匹配上
            unmatched_detections.append(detection_idx)                                          # 表示当前检测未匹配上
        else:
            matches.append((track_idx, detection_idx))                                          # 匹配上的对象

    return matches, unmatched_tracks, unmatched_detections




# 级联匹配
def matching_cascade(distance_metric, max_distance, cascade_depth, tracks, 
                        detections, track_indices=None, detection_indices=None):
    
    # 我们的级联匹配是要匹配所有confirmed的跟踪目标, 假如要匹配所有的跟踪目标, 设置track_indices=None
    # 因此，这里传进来的track_indices是confirmed_tracks
    if track_indices == None:                                       # 默认将所有的跟踪对象作为匹配对象
        track_indices = list(range(len(tracks)))
    if detection_indices == None:                                   # 默认将所有的检测目标作为匹配对象
        detection_indices = list(range(len(detections)))
    
    unmatched_detections = detection_indices                        # 未匹配过的检测目标的索引
    matches = []    
      
    # 根据层级深度进行级联匹配
    # 层级越深,表示跟踪目标丢失的帧数越多,应该越晚匹配
    # 层级越浅,表示最近的跟踪目标, 丢失帧数少,应该越早匹配
    for level in range(cascade_depth):
        if len(unmatched_detections) == 0:                          # 没有未匹配的检测对象
            break
        
        track_indices_l = [k for k in track_indices if tracks[k].time_since_update == 1 + level]       # 当前层级的跟踪对象
        if len(track_indices_l) == 0:                                                           # 如果当前层级没有跟踪对象,继续下一层级
            continue
        
        # 最小代价匹配, 先执行外观匹配, 再执行马氏距离限制代价矩阵
        matches_l, _, unmatched_detections = min_cost_matching(distance_metric, max_distance, tracks, detections,
                                                                    track_indices_l, unmatched_detections)

        matches += matches_l                                                                    # 存储匹配的跟踪目标和检测目标的索引
    
    unmatched_tracks = list(set(track_indices_l) - set(k for k, _ in matches))                  # 未匹配上的跟踪目标的索引
    
    return matches, unmatched_tracks, unmatched_detections



# 通过马氏距离限制修改代价矩阵
# 门控代价矩阵
# 门控代价矩阵的作用就是通过计算卡尔曼滤波的状态分布和测量值之间的马氏距离对代价矩阵进行限制。
# 代价矩阵中的距离是Track和Detection之间的表观相似度，假如一个轨迹要去匹配两个表观特征非常相似的Detection，
# 这样就很容易出错，但是这个时候分别让两个Detection计算与这个轨迹的马氏距离，并使用一个阈值gating_threshold进行限制，
# 所以就可以将马氏距离较远的那个Detection区分开，可以降低错误的匹配。
def gate_cost_matrix(kf, cost_matrix, tracks, detections, track_indices, detection_indices,
                     gated_cost=INFTY_COST, only_position=False):
    
    # 根据通过卡尔曼滤波获得的状态分布，使成本矩阵中的不可行条目无效。
    gating_dim = 2 if only_position else 4                              # 卡方校验的自由度
    gating_threshold = kalman_filter.chi2inv95[gating_dim]              # 9.4877
    
    measurements = np.asarray([detections[i].to_xyah() for i in detection_indices])     # 将检测结果转换为测量值
    
    for row, track_idx in enumerate(track_indices):                                     # 遍历每一个跟踪对象
        track = tracks[track_idx]                                                       # 当前的一个跟踪对象
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)                  # 计算跟踪对象与所有检测对象的马氏距离
        
        # 当马氏距离超过卡方检验的值时, 表示将两者匹配犯错误的概率较高, 因此将代价修改为较大的值, 表示不匹配它们
        cost_matrix[row, gating_distance > gating_threshold] = gated_cost               # 当马氏距离超过阈值,代价矩阵设置为inf
    
    return cost_matrix



































