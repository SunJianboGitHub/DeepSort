#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   deepsort_sample.py
@Time    :   2023/02/23 14:25:50
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   简单多目标跟踪-带有卡尔曼滤波
'''

import cv2
import zmq
import math
import numpy as np

import scipy.linalg
from scipy.optimize import linear_sum_assignment            # 匈牙利匹配算法


from preprocess import *


chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}



class KalmanFilter:
    def __init__(self):
        ndim, dt= 4, 1.0
        
        # x` = Fx
        # F = self._motion_mat
        #  1       1
        #    1       1
        #      1       1
        #        1       1
        #          1
        #            1
        #              1
        #                1
        #
        #  F = {
        #   1,      dt
        #   0,      1
        #  }
        # x = [cx, cy, a, h, vcx, vcy, va, vh].T
        # x` = Fx
        # cx` = cx + vcx
        # cy` = cy + vcy
        # a` = a + va
        # h` = h + vh
        # vcx' = vcx
        # vcy` = vcy
        # va` = va
        # vh` = vh
        
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)               # 这里应该是指状态转移矩阵，维度是8*8
        for i in range(ndim):                                       # 这里设置了目标匀速直线运动
            self._motion_mat[i, ndim + i] = dt
        
        # H，测量函数，测量矩阵
        # H 是将x从状态空间转换到测量空间，然后计算残差。或者说是映射函数，维度是4*8
        self._update_mat = np.eye(ndim, 2*ndim)

        self._std_weight_position = 1./20
        self._std_weight_velocity = 1./160
        
    
    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]

        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]

        # 运动的协方差，Q
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # x` = Fx
        mean = np.dot(self._motion_mat, mean)

        # P = FPF.T + Q
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T
        )) + motion_cov
        return mean, covariance

    
    def project(self, mean, covariance):
        # 把输入映射到测量空间
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]

        # R，测量噪声的协方差矩阵
        innovation_cov = np.diag(np.square(std))

        # H @ x
        mean = np.dot(self._update_mat, mean)

        # S = HPH.T + R
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T
        )) + innovation_cov
        return mean, covariance
    
    
    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )

        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False
        ).T
 
        # 计算残差 y = z - Hx
        innovation = measurement - projected_mean

        # 计算新的估计值 x` = x + Ky
        new_mean = mean + np.dot(innovation, kalman_gain.T)

        # 计算的是估计值的协方差 P = P - KHP
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T
        ))
        return new_mean, new_covariance
    
    def gating_distance(self, mean, covariance, measurements, only_position=False):
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        # 矩阵分解：巧克力分解, y = LL.T
        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        
        # z = L^-1 d
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, overwrite_b=True, check_finite=False
        )
        # 服从4个自由度的卡方分布
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
    

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        
        # 把两个向量按行拼接起来，其实这里是拼接为1*8
        mean = np.r_[mean_pos, mean_vel]
        
        # 定义标准差
        std = [
            # 位置标准差
            2 * self._std_weight_position * measurement[3],            # center_x的初始标准差
            2 * self._std_weight_position * measurement[3],            # center_y的初始标准差
            1e-2,                                                      # 宽高比的初始标准差
            2 * self._std_weight_position * measurement[3],            # height的初始标准差
            
            # 速度标准差
            10 * self._std_weight_velocity * measurement[3],            # center_x的初始标准差
            10 * self._std_weight_velocity * measurement[3],            # center_y的初始标准差
            1e-5,                                                       # 宽高比的初始标准差
            10 * self._std_weight_velocity * measurement[3],            # height的初始标准差
        ]
        
        covariance = np.diag(np.square(std))

        return mean, covariance                                         # 就是卡尔曼滤波中的x, P
    
    
    

class State:
    Tentative = 1                                           # 待定、暂定
    Confirmed  = 2                                           # 确认的
    Deleted   = 3                                           # 删除的



class Track:
    def __init__(self, detect, mean, covariance, id):
        self.id = id
        self.mean = mean
        self.covariance = covariance
        self.trace = [detect]
        self.last_position = detect
        self.time_since_update = 0
        self.hits = 0
        self.state = State.Tentative
        pass

    def predict(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.time_since_update += 1
        pass
    
    
    def update(self, kf, detect):
        # 匹配上的track才会执行update
        self.trace.append(detect)
        
        # 如果轨迹太长，则删除掉第一个
        if len(self.trace) > 50:
            self.trace.pop(0)
        
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection_to_xyah(detect))
        self.last_position = detect
        self.hits += 1
        self.time_since_update = 0
        
        # 如果当前状态为待定，并且击中了3次（匹配了3帧的框）,
        # 则可以认为是一个鲁棒的目标转为确认
        if self.state == State.Tentative and self.hits >= 3:
            self.state = State.Confirmed
        

    def mark_missed(self):
        # 如果状态是待定的，则表示没有用了
        if self.state == State.Tentative:
            self.state = State.Deleted
            
        # 如果状态是确认的，并且距离最新时间超过30帧，也就是丢失了30帧
        # 表示这个对象消失了
        elif self.time_since_update > 30 and self.state == State.Confirmed:
            self.state = State.Deleted
        
        
        
        
        pass





class Tracker:
    def __init__(self):
        
        # 初始化目标跟踪器
        self.id_next = 1                                     # ID 计数器,如果需要ID,则这个变量累加
        self.tracks = []                                     # 存储所有的目标跟踪对象
        
        self.kf = KalmanFilter()                             # 创建卡尔曼滤波
        
        pass

    # 预测环节,对应卡尔曼滤波的predict
    def predict(self):
        for track in self.tracks:
            track.predict(self.kf)
        
        
        pass

    # 更新环节
    def update(self, detections):
        
        unmatched_tracks_idx  = np.arange(len(self.tracks))                          # 给所有的跟踪目标指定索引
        unmatched_detects_idx = np.arange(len(detections))                           # 给所有的检测目标指定索引
        
        nlevel = 30                                                                  # 级联匹配的深度
        filter_states = [State.Confirmed, State.Tentative]                           # 匹配优先级
        
        
        # 实现了优先匹配确认的，然后匹配待定的，不匹配删除的
        # 实现时间距离更近的先匹配
        for state in filter_states:
            for ilevel in range(nlevel):
                if len(unmatched_tracks_idx) == 0 or len(unmatched_detects_idx) == 0:
                    break
                
                # 挑选tracks，必须满足：1-状态是要求的，2-time_since_update匹配的（距离更新的时间间隔）
                # time_since_update距离更新的时间间隔
                select_tracks_idx = np.array([index for index in unmatched_tracks_idx
                                     if self.tracks[index].time_since_update == ilevel + 1 and 
                                     self.tracks[index].state == state], dtype=int)
                
                # 如果没有选中的tracks，则直接下一步
                if len(select_tracks_idx) == 0:
                    continue
                    
                    
                matched_tracks_idx, matched_detects_idx = self.match(
                                            select_tracks_idx, unmatched_detects_idx, detections)
        
        
                # 去掉匹配上的detections和tracks
                unmatched_tracks_idx  = np.array(list(set(unmatched_tracks_idx) - set(matched_tracks_idx)))
                unmatched_detects_idx = np.array(list(set(unmatched_detects_idx) - set(matched_detects_idx)))
                for i_track, i_detect in zip(matched_tracks_idx, matched_detects_idx):
                    self.tracks[i_track].update(self.kf, detections[i_detect])
                    
        
        # 处理未匹配的tracks，这里可能出现delete状态，那么删除它
        for i_track in unmatched_tracks_idx:
            self.tracks[i_track].mark_missed()
            
        # 移除状态为删除的目标
        self.tracks = [track for track in self.tracks if track.state != State.Deleted]
            
        # 处理未匹配的detect
        for i_detect in unmatched_detects_idx:
            self.make_new_track(detections[i_detect])
        
        pass
    
    def make_new_track(self, detect):
        
        mean, covariance = self.kf.initiate(detection_to_xyah(detect))
        
        new_track_id = self.id_next
        self.tracks.append(Track(detect, mean, covariance, new_track_id))
        
        
        
        self.id_next += 1
        
        pass
    
    
    # 匹配函数,实现tracks和detections之间的匹配
    def match(self, tracks_index, detections_index, detections):
        # tracks_index，指定为track在self.tracks中的索引
        # detections_index，指定为detection在detections中的索引
        # 因为是集合操作，所以传索引比较方便
        # 这里的匹配是选中其中部分进行的，不是全部
        # 因为不同情况权重优先级不同
        # 返回值 matched_detections, matched_tracks
        
        INFTY_COST = 1e+5
        gating_threshold_95 = chi2inv95[4]
        num_track = len(tracks_index)                                               # 当前需要匹配的跟踪目标数
        num_detection = len(detections_index)                                       # 当前需要匹配的检测目标数
        cost_matrix = np.zeros((num_track, num_detection), dtype=np.float32)        # 初始化代价矩阵
        
        # 设置代价矩阵
        for row, i_track in enumerate(tracks_index):
            for col, i_detection in enumerate(detections_index):
                track     = self.tracks[i_track]                                    # 当前的跟踪目标
                detection = detections[i_detection]                                 # 当前的检测目标
                
                # 如果track与detection之间的马氏距离无法通过卡方检验，则这里设置cost为极大值
                measurement = detection_to_xyah(detection)
                gating_distance = self.kf.gating_distance(
                    track.mean, track.covariance, measurement, only_position=False
                )

                if gating_distance > gating_threshold_95:
                    # 校验不通过
                    cost_matrix[row, col] = INFTY_COST
                else:
                    # 计算track与detection之间的代价
                    cost_matrix[row, col] = distance(track.last_position, detection)
                
                pass
        center_distance_threshold = 100
        matched_table = cost_matrix < center_distance_threshold
        
        rows, cols = linear_sum_assignment(cost_matrix)
        cost_matched_table = np.zeros((num_track, num_detection), dtype=bool)
        cost_matched_table[rows, cols] = True
        
        matched_tracks, matched_detections = np.where(matched_table & cost_matched_table)
        
        return tracks_index[matched_tracks], detections_index[matched_detections]





def distance(a, b):
    # a和b是两个detection[1x5]
    # 计算两个中心的距离
    # 这个计算考虑很easy，如果尺度大小差异很大，这个计算不合理
    aleft, atop, aright, abottom = a[:4]
    bleft, btop, bright, bbottom = b[:4]
    x1 = (aleft + aright) * 0.5
    y1 = (atop + abottom) * 0.5
    x2 = (bleft + bright) * 0.5
    y2 = (btop + bbottom) * 0.5
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)














if __name__ == "__main__":
    image_dir      = "PETS09-S2L1/img1"                             # 图像的目录
    detection_file = "PETS09-S2L1/det/det.txt"                      # 目标检测结果文件路径
    annotations = load_detection_annotations(detection_file)        # 加载目标检测结果

    # 获取图像ID,并去重. 目标检测结果的第0个位置表示图像的ID
    image_ids = np.unique(annotations[:, 0]).astype(int)
    
    tracker = Tracker()
    random_color = (np.random.rand(32, 3) * 255).astype(int)
    zmq_server = ZmqShow(port=12345)
    
    for img_id in image_ids:
        
        # 获取第i张图像的检测结果
        # detections.shape = n x 10
        # 格式定义为：
        # https://motchallenge.net/data/MOT15/
        # https://github.com/dendorferpatrick/MOTChallengeEvalKit
        # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        detections = annotations[annotations[:, 0] == img_id]           # 维度是 [n, 10]
        detections = detections[:, [2, 3, 4, 5, 6]]                     # 只保留[x, y, w, h, conf], 维度是 [n, 5]

        # 转置后解包,类似于yolov5的操作
        left, top, width, height = detections[:, :4].T
        right  = left + width - 1                                       # 计算检测框右下角坐标
        bottom = top + height - 1
        detections[:, 2] = right
        detections[:, 3] = bottom
        detections       = nms(detections, 0.25, 4)                      # 非极大值抑制, NMS操作
        # 至此, detections内容变为 [x, y, r, b, conf], 维度是 [n, 5]
        
        # 开始跟踪
        tracker.predict()
        tracker.update(detections)
        
        image_file = f"{image_dir}/{img_id:06d}.jpg"
        image = cv2.imread(image_file)
        cv2.putText(image, f"{img_id}", (10, 50), 0, 2, (0, 255, 0), 2, 16, False)
        
        for track in tracker.tracks:
            location = [int(item) for item in track.last_position]
            track_color = random_color[track.id % random_color.shape[0]]
            if track.state == State.Confirmed and track.time_since_update == 0:
                draw_box(image, location[:2], location[2:4], color=track_color, text=f"{track.id}", trace=track.trace)
        
        
        
        img_data = cv2.imencode(".jpg", image)[1].tobytes()
        
        zmq_server.send(img_data)
        print(f"Process: {img_id}")
        

























