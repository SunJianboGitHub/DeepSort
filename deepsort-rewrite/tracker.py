#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   tracker.py
@Time    :   2023/02/23 23:26:08
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   多目标跟踪器
'''


from __future__ import absolute_import
import numpy as np
import kalman_filter
import linear_assignment
import iou_matching

from track import Track



class Tracker:
    '''
    这是一个多目标跟踪器
    
    参数
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        测量和跟踪之间的距离矩阵
        
    max_age : int
        在跟踪对象的状态被设置为`Deleted`之前, 跟踪对象被连续丢失的最大次数。
        
    n_init : int
        在跟踪状态被设置为`Confirmed`之前, 连续检测匹配的帧数(比如设置为3)。如果`Tentative`目标在
        第一个`n_init` 帧之内发生了丢失, 则将跟踪对象的状态设置为`Deleted`。

    属性
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        测量和跟踪之间的距离矩阵
        
    max_age : int
        在跟踪对象的状态被设置为`Deleted`之前, 跟踪对象被连续丢失的最大次数。
        
    n_init : int
        在跟踪状态被设置为`Confirmed`之前, 连续检测匹配的帧数(比如设置为3)。如果`Tentative`目标在
        第一个`n_init` 帧之内发生了丢失, 则将跟踪对象的状态设置为`Deleted`。
        
    kf : kalman_filter.KalmanFilter
        卡尔曼滤波器，用于过滤图像空间中的目标轨迹。
        
    tracks : List[Track]
        跟踪目标对象列表。
    
    '''

    def __init__(self, metric, max_iou_distance=0.3, max_age=70, n_init=3):
        self.metric = metric                                        # 距离度量方式
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age                                      # 最后一次测量更新后, 连续丢失帧数超过最大年纪,状态变为Deleted
        self._n_init = n_init                                       # 连续击中的次数, 超过它, 确认为跟踪目标, 状态从暂定到确认    
        
        self.kf = kalman_filter.KalmanFilter()                      # 多个目标共用一个卡尔曼滤波器
        self.tracks = []                                            # 存储跟踪目标
        self._next_id = 1                                           # 跟踪目标的唯一ID


    def predict(self):
        '''
        每一个时间步, 前向传递跟踪对象的状态变量分布。在每次update之前, 该函数应该被调用。
        '''
        for track in self.tracks:                                   # 遍历每一个跟踪目标, 包括确认的和暂定的
            track.predict(self.kf)                                  # 根据前一步的状态和过程模型, 计算先验估计
    
    
    def update(self, detections):
        '''
        执行测量更新和跟踪管理
        
        参数
        ----------
        detections : List[deep_sort.detection.Detection]
            当前时间步的检测目标列表
        '''
        
        # 运行级联匹配和IOU匹配
        matches, unmatched_tracks, unmatched_detections = self._match(detections)
        
        # 对匹配上的tracks，进行测量更新
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])
        
        # 对于未匹配上的tracks, 执行标记删除
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
        
        # 对于未匹配的检测, 初始化为新的跟踪对象
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        
        # 更新距离度量
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]                  # 确认的跟踪对象的唯一ID
        features, targets = [], []
        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features                                              # 存储所有的跟踪对象的所有的feature
            targets += [track.track_id for _ in track.features]                     # 每个feature对应的track_id
            track.features = []                                                     # 将跟踪对象的特征设置为空列表
        
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)              # 只对确认的对象进行外观匹配
        
        
    
    def _match(self, detections):
        '''
        这里先执行级联匹配, 再进行IoU匹配。
        
        '''
        def gated_metric(tracks, detections, track_indices, detection_indices):
            features =np.array([detections[i].feature for i in detection_indices])                      # 所有未匹配的检测目标的外观特征向量
            targets = np.array([tracks[i].track_id for i in track_indices])                             # 未匹配的跟踪对象的唯一ID
            
            cost_matrix = self.metric.distance(features, targets)                                       # 外观相似度距离
            cost_matrix = linear_assignment.gate_cost_matrix(self.kf, cost_matrix, tracks, detections, 
                                                             track_indices, detection_indices)          # 通过马氏距离限制修改代价矩阵
            
            return cost_matrix
        
        # 分割跟踪集合为confirmed和unconfirmed tracks
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]           # 确认的跟踪目标索引
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]     # 未确认的跟踪目标索引
        
        # 级联匹配
        # 使用外观纹理特征和马氏距离, 关联匹配跟踪对象
        matches_a, unmatched_tracks_a, unmatched_detections = linear_assignment.matching_cascade(
                                                    gated_metric, self.metric.matching_threshold, self.max_age,
                                                    self.tracks, detections, confirmed_tracks)
        
        # IoU匹配
        # 使用IoU将剩余的confirmed tracks和unconfirmed tracks与未匹配的detections进行IoU关联匹配
        # IoU 匹配的跟踪包括：不确定的目标 + 确认但是未丢失的目标
        # 切记, 并不是所有的未匹配的确认对象都进行IoU匹配, 只是对未丢失过的进行匹配。丢失太久只靠IoU匹配, 很容易匹配错误
        # 因此, 需要下一轮进行外观特征匹配
        iou_track_candidates = unconfirmed_tracks + [k for k in unmatched_tracks_a
                                                     if self.tracks[k].time_since_update == 1]              # 只对最近未丢失的确认跟踪目标执行IoU匹配
        unmatched_tracks_a = [k for k in unmatched_tracks_a if self.tracks[k].time_since_update != 1]       # 很久未匹配上的目标不执行IoU匹配, 再下一轮执行外观匹配,减少ID切换
        
        matches_b, unmatched_tracks_b, unmatched_detections = linear_assignment.min_cost_matching(
                            iou_matching.iou_cost, self.max_iou_distance, self.tracks, detections,
                            iou_track_candidates, unmatched_detections)                                     # 执行IoU匹配
        
        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        
        
        return matches, unmatched_tracks, unmatched_detections


    
    
    def _initiate_track(self, detection):
        '''
        针对未匹配上的检测目标, 构建新的跟踪对象。新的跟踪对象需要初始化状态变量均值和协方差
        '''
        mean, covariance = self.kf.initiate(detection.to_xyah())                        # 将检测结果设置为初始状态,并根据高度参数设置协方差矩阵
        self.tracks.append(Track(mean, covariance, self._next_id, 
                                 self._n_init, self.max_age, detection.feature))        # 构建新的跟踪对象, 并将其加入跟踪列表
        self._next_id += 1                                                              # 自增跟踪对象的唯一ID
        

        



























































