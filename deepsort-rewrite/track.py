#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   track.py
@Time    :   2023/02/23 20:58:03
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   创建一个跟踪对象
'''



# 状态枚举类型，用于记录跟踪对象的状态
# 状态包括三种类型：暂定、确认、删除
class TrackState:
    '''
    单个目标跟踪的状态是枚举类型。在收集到足够的证据之前(hits > 3), 新创建的跟踪目标被归类为"暂定"。
    然后, 跟踪对象状态变为"确认"。不再存在的跟踪被分类为"删除", 已将其标记为准备从跟踪列表中删除。
    '''
    
    Tentative = 1                                               # 暂定状态, 当hit > 3时, 变为确认状态
    Confirmed = 2                                               # 确认状态, 当丢失超过30帧时, 变为删除状态
    Deleted   = 3                                               # 删除状态, 将当前跟踪对象从跟踪列表中删除



# 跟踪对象类
class Track:
    '''
    单个目标跟踪采用的状态变量是`(x, y, a, h)` 和它们对应的速度。其中`(x, y)`是边界框的中心点, `a` 是
    宽高比(aspect ratio),  `h` 是边界框的高度。
    
    参数
    ----------
    mean : ndarray
        状态变量的初始值, 分别是[x, y, a, h, vx, vy, va, vh]。类似于卡尔曼滤波中的 x
        
    covariance : ndarray
        状态变量初始值对应的协方差矩阵, 也是初始值的不确定性。类似于卡尔曼滤波中的 P
        
    track_id : int
        跟踪对象的唯一ID
        
    n_init : int
        在跟踪状态被设置为`Confirmed`之前, 连续检测匹配的帧数(比如设置为3)。如果`Tentative`目标在
        第一个`n_init` 帧之内发生了丢失, 则将跟踪对象的状态设置为`Deleted`。
        
    max_age : int
        在跟踪对象的状态被设置为`Deleted`之前, 跟踪对象被连续丢失的最大次数。
        
    feature : Optional[ndarray]
        此跟踪来自于检测的特征向量。如果不是 None, 这个特征将会被添加到 `features` 缓存中。
    
    属性
    ----------
    mean : ndarray
        状态变量的初始值, 分别是[x, y, a, h, vx, vy, va, vh]。类似于卡尔曼滤波中的 x
        
    covariance : ndarray
        状态变量初始值对应的协方差矩阵, 也是初始值的不确定性。类似于卡尔曼滤波中的 P
        
    track_id : int
        跟踪对象的唯一ID
        
    hits : int
        测量更新的总次数, 也就是跟踪目标匹配的总次数
        
    age : int
        自第一次出现以来的总帧数。
        
    time_since_update : int
        自从最后一次测量更新后的总帧数。
        
    state : TrackState
        当前跟踪对象的状态。
        
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.
    
    '''

    def __init__(self, mean, covariance, track_id, n_init, max_age, feature=None):
        self.mean = mean                                    # 跟踪对象的初始状态, 一般是第一次的检测结果, 用 x 表示
        self.covariance = covariance                        # 跟踪对象初始状态的协方差, 一般用 P 表示
        self.track_id = track_id                            # 当前跟踪对象的唯一ID
        self.hits = 1                                       # 目标被击中的次数
        self.age = 1                                        # 自第一次出现以来的总帧数
        self.time_since_update = 0                          # 最后一次测量更新后得总帧数, 也就是丢失的帧数
        
        self.state = TrackState.Tentative                   # 第一次构建跟踪对象时, 它的状态是暂定的
        self.features = []                                  # 存储跟踪对象的外观纹理特征, 用于匹配
        if feature is not None:                             # 如果存在纹理特征
            self.features.append(feature)                   # 添加到特征列表
        
        self._n_init = n_init                               # 连续击中的次数, 超过它, 跟踪对象的状态才可以从暂定->确认
        self._max_age = max_age                             # 在跟踪对象状态设置为`Deleted`之前, 连续丢失帧的最大数量
    
    
    
    def predict(self, kf):
        '''
        使用卡尔曼滤波器的预测步骤, 去传递当前时间步的状态变量的分布。
        
        参数
        ----------
        kf : kalman_filter.KalmanFilter
            卡尔曼滤波器
        '''
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)             # 根据初始位置和过程模型, 计算先验估计
        self.age += 1                                                                   # 统计目标自第一次出现的总帧数, 和max_age关联不大
        self.time_since_update += 1                                                     # 最后一次测量更新后, 目标连续丢失的帧数
        
    
    
    def update(self, kf, detection):
        '''
        执行卡尔曼滤波器的测量更新步骤, 并且更新纹理特征列表。
        
        参数
        ----------
        kf : kalman_filter.KalmanFilter
            卡尔曼滤波器。
        detection : Detection
            相关检测。
        '''
        
        self.mean, self.covariance = kf.update(
                    self.mean, self.covariance, detection.to_xyah())                   # 卡尔曼滤波的更新步骤, 也就是后验估计
        self.features.append(detection.feature)                                        # 记录新的纹理特征
        
        self.hits += 1                                                                 # 记录当前目标击中的次数
        self.time_since_update = 0                                                     # 重置连续丢失的帧数, 测量更新表示当前帧未丢失
        
        if self.state == TrackState.Tentative and self.hits >= self._n_init:           # 达到连续击中的次数, 暂定变为确认
            self.state = TrackState.Confirmed
        pass


    def mark_missed(self):
        # 对于未匹配到的跟踪对象, 包括两种：确认的和暂定的
        # 如果跟踪对象是确认的, 并且连续都是帧数超过 max_age, 设置其状态为 Deleted
        if self.state == TrackState.Confirmed and self.time_since_update > self._max_age:
            self.state = TrackState.Deleted
        
        # 如果跟踪对象是暂定的
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        
    
    
    def is_tentative(self):
        '''如果跟踪对象的状态是暂定的(unconfirmed), 返回True'''
        return self.state == TrackState.Tentative
    
    def is_confirmed(self):
        '''如果跟踪对象的状态是确认的(confirmed), 返回True'''
        return self.state == TrackState.Confirmed
    
    def is_deleted(self):
        '''如果跟踪对象的状态是删除的(deleted), 返回True'''
        return self.state == TrackState.Deleted



    def to_tlwh(self):
        ret = self.mean[:4].copy()                      # 这里应该是xyah
        ret[2] *= ret[3]                                # 将 a -> w
        ret[:2] -= ret[2:] / 2                          # 转换之后变为 tlwh
        
        return ret

    def to_tlbr(self):
        ret = self.to_tlwh()
        ret[2:] = ret[2:] + ret[:2]

        return ret










































