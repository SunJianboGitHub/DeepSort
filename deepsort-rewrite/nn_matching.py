#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   nn_matching.py
@Time    :   2023/02/25 23:52:14
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   None
'''

import numpy as np


'''
该文件的主要功能就是：
    1. 计算向量直接的距离: 欧氏距离、余弦距离。
    2. 通过外观的距离, 计算代价矩阵
'''





# 欧式距离
def _pdist(a, b):
    '''
    用于计算成对的平方距离:
       1. a NxM 代表N个对象, 每个对象有M个数值作为embedding进行比较
       2. b LxM 代表L个对象, 每个对象有M个数值作为embedding进行比较 
       3. 返回的是NxL的矩阵, 比如dist[i][j]代表a[i]和b[j]之间的平方和距离
       4. 实现见: https://blog.csdn.net/frankzd/article/details/80251042
    '''
    
    a, b = np.asarray(a), np.asarray(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    
    a2, b2 = np.square(a).sum(sxis=1), np.square(b).sunm(axis=1)
    r2 = a2[:, None] + b2[None, :] - 2 * np.dot(a, b.T)
    r2 = np.clip(r2, 0, float(np.inf))
    
    return r2


# 余弦距离
def _cosine_distance(a, b, data_is_normalized=False):
    '''
    用于计算成对的余弦距离:
        1. a : [NxM] b : [LxM]
        2. 余弦距离 = 1 - 余弦相似度
        3. 实现见: https://blog.csdn.net/u013749540/article/details/51813922
    '''
    # 需要将余弦相似度转化成类似欧氏距离的余弦距离。
    # np.linalg.norm 操作是求向量的范式，默认是L2范式，等同于求向量的欧式距离。
    if not data_is_normalized:
        a = np.asarray(a) / np.linalg.norm(a, axis=1, keepdims=True)
        b = np.asarray(b) / np.linalg.norm(b, axis=1, keepdims=True)
    
    return 1 - np.dot(a, b.T)
    

# 返回最近的欧式距离
def _nn_euclidean_distance(x, y):
    
    distances = _pdist(x, y)                            # 维度是[N, L]
    min_dis = distances.min(axis=0)                     # 在行的维度上求最小值,也就是第一列的最小值
    
    return np.maximum(0.0, min_dis)                     # 返回最近的欧式距离


# 返回最近的余弦距离
def _nn_cosine_distance(x, y):
    
    distances = _cosine_distance(x, y)                  # 余弦距离, 维度是[N, L]
    min_dis = distances.min(axis=0)                     # 在行的维度上求最小值,也就是第一列的最小值

    return min_dis                                      # 返回最近的余弦距离




# 对每个目标, 返回一个最近的距离
class NearestNeighborDistanceMetric(object):
    """
    
    A nearest neighbor distance metric that, for each target, returns
    the closest distance to any sample that has been observed so far.

    参数
    ----------
    metric : str
        "euclidean" 或者 "cosine"距离度量标准
    matching_threshold: float
        匹配阈值。拥有较大距离的样本被考虑为无效的匹配。
    budget : Optional[int]
        If not None, fix samples per class to at most this number. Removes
        the oldest samples when the budget is reached.

    Attributes
    ----------
    samples : Dict[int -> List[ndarray]]
        A dictionary that maps from target identities to the list of samples
        that have been observed so far.

    """
    def __init__(self, metric, matching_threshold, budget=None):
        if metric == "euclidean":                                                          # 使用欧氏距离寻找最近邻
            self._metric = _nn_euclidean_distance
        elif metric == "cosine":                                                           # 使用余弦距离寻找最近邻
            self._metric = _nn_cosine_distance
        else:
            raise ValueError("Invalid metric; must be either 'euclidean' or 'cosine'")
        
        self.matching_threshold = matching_threshold                                       # 在级联匹配中调用
        self.budget = budget                                                               # 预算, 控制feature的多少
        self.samples = {}                                                                  # samples是一个字典{id->feature list}
            
        pass

    
    def partial_fit(self, features, targets, active_targets):
        """
        使用新数据更新距离度量

        参数
        ----------
        features : ndarray
            一个 NxM 矩阵, N 个特征, 特征的维度是 M。
        targets : ndarray
            一个整型数组, 关联 target 身份.
        active_targets : List[int]
            A list of targets that are currently present in the scene.
        """
        
        for feature, target in zip(features, targets):                              # 遍历每一个特征以及它对应的ID
            self.samples.setdefault(target, []).append(feature)                     # 每个ID对应一个二维数组,存储feature
            if self.budget is not None:                                             # 如果特征预算不是空
                self.samples[target] = self.samples[target][-self.budget:]          # 只需要指定数目的特征, 从后往前保留
        
        # 筛选激活的目标
        self.samples = {k: self.samples[k] for k in active_targets}
        
    
    def distance(self, features, targets):
        '''
        计算检测目标的features和跟踪对象targets存储的序列特征的距离(相似度)
        
        参数
        ----------
        features : ndarray
            一个 NxM 矩阵, N 表示检测目标数目, 特征的维度是 M.
        targets : List[int]
            一个跟踪对象的列表, 去和检测目标匹配。存储的是跟踪对象的ID
            
        返回值
        -------
        ndarray
            返回一个代价矩阵, 形状是 len(targets), len(features), 其中element (i, j) 包含了 `targets[i]` 和 `features[j]`最近的距离。
        '''
        
        cost_matrix = np.zeros((len(targets), len(features)))                                   # 根据维度,初始化代价矩阵
        for i, target in enumerate(targets):                                                    # 遍历每一个跟踪对象, 其中target是跟踪对象的ID
            # 返回的是当前跟踪对象与所有检测目标特征的最小距离
            # 切记当特征为空时，features=[nan, nan, ...]
            # 但是np.nan == np.nan结果为FALSE
            if features[0] == features[0]:
                cost_matrix[i, ] = self._metric(self.samples[target], features)                     # self.samples[target]可以获得当前跟踪对象的所有外观特征     
            pass
        return cost_matrix
        
        
        





































