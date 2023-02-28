#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   detection.py
@Time    :   2023/02/24 11:07:33
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   处理检测目标
'''

import numpy as np


class Detection(object):
    '''
    这个类代表了: 单张图像的所有检测边界框。

    参数
    ----------
    tlwh : array_like
        边界框的格式 `(top left x, top left y, w, h)`
        
    confidence : float
        检测边界框的置信度得分
        
    feature : array_like
        在图像中, 描述被包含对象的特征向量

    属性
    ----------
    tlwh : ndarray
        边界框的格式 `(top left x, top left y, w, h)`
        
    confidence : float
        检测边界框的置信度得分
        
    feature : array_like
        在图像中, 描述被包含对象的特征向量
    '''
    
    def __init__(self, tlwh, confidence, feature):
        self.tlwh = np.asarray(tlwh, dtype=float)                       # 检测结果, 格式为(top left x, top left y, width, height)
        self.confidence = float(confidence)                             # 边界框的置信度得分
        self.feature = np.asarray(feature, dtype=np.float32)            # 边界框的特征向量
        
        

    def to_tlbr(self):
        '''
        转换边界框格式: tlwh -> tlbr
        '''
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        
        return ret
    
    def to_xyah(self):
        '''
        转换边界框格式为 `(center x, center y, aspect ratio, height)`, 其中 aspect ratio 是 `width / height`.
        '''
        ret = self.tlwh.copy()                              # 拷贝一份, 防止污染原始数据
        ret[:2] += (ret[2:] / 2)                             # 计算中心点坐标
        ret[2] /= ret[3]                                    # 计算宽高比
        return ret




























































