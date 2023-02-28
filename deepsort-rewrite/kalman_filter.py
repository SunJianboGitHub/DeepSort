#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   kalman_filter.py
@Time    :   2023/02/24 00:13:16
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   None
'''

import scipy
import numpy as np

"""
N自由度卡方分布0.95分位数的表(包含N=1, 2, 3, ...9), 取自MATLAB/Octave的chi2v函数, 并用作Mahalanobis门控距离阈值。
"""
# 用于马氏距离的卡方校验
# 卡方检验就是统计样本的实际观测值与理论推断值之间的偏离程度，实际观测值与理论推断值之间的偏离程度就决定卡方值的大小，
# 如果卡方值越大，二者偏差程度越大；反之，二者偏差越小；若两个值完全相等时，卡方值就为0，表明理论值完全符合。
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




class KalmanFilter(object):
    """
    一个简单的卡尔曼滤波器,用于在图像空间中跟踪边界框。

    8-dimensional 状态空间

        x, y, a, h, vx, vy, va, vh

    包含了边界框中心点的位置 (x, y), 宽高比 a, 高度 h,和它们各自的速度。
    
    目标运动采用匀速模型。边界框位置 (x, y, a, h) 是可以被直接观测的 (线性观测模型)。
    """
    def __init__(self, ndims=4, dt=1.0):
        
        # 构建过程模型, 创建状态转移矩阵, 假设目标是匀速直线运动
        self.F = np.eye(2*ndims, 2*ndims)
        for i in range(ndims):
            self.F[i, i+ndims] = dt
        
        # 构建测量函数, 将状态变量从状态空间转换到测量空间
        self.H = np.eye(ndims, 2*ndims)
        
        # 相对于当前状态估计, 选择运动和观测的不确定性。
        # 在系统模型中,这些权重控制着不确定性的大小,这有点奇怪。
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160
    
    def initiate(self, measurement):
        '''
        第一次创建跟踪对象时, 需要初始化状态变量和协方差, 即 x 和 P
        
        参数
        ----------
        measurement : ndarray
            边界框坐标 (x, y, a, h), 其中位置 (x, y), 宽高比 a, 高度 h。

        返回
        -------
        (ndarray, ndarray)
            返回先验估计的均值和协方差。为观测的速度均值被初始化为0。
        '''
        pos = measurement                                           # 跟踪框的初始值, 也就是检测的结果
        vel = np.zeros_like(pos)                                    # 不可观测的速度的初始值设置为 0
        x = np.r_[pos, vel]                                         # 拼接为状态变量, 维度是 8*1      
        
        p_std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        
        P = np.diag(np.square(p_std))                               # 状态变量之间的协方差矩阵
        
        return x, P
    
    
    def predict(self, x, P):
        """
        运行卡尔曼滤波器的预测步骤。
        
        参数
        ----------
        x : ndarray
            上一个时间步长, 状态变量的均值(它是一个8维向量)
            
        P : ndarray
            上一个时间步长, 状态变量之间的协方差矩阵(它是一个8*8矩阵)

        返回值
        -------
        (ndarray, ndarray)
            返回先验估计的均值和协方差。为观测的速度均值被初始化为0。
        """
        
        # 根据前一个状态的后验估计, 构造过程噪声矩阵Q
        # x[3]表示边界框的高度, 高度越高, 移动越快,方差越大
        std_pos = [
            self._std_weight_position * x[3],
            self._std_weight_position * x[3],
            1e-2,
            self._std_weight_position * x[3]]
        
        std_vel = [self._std_weight_velocity * x[3],
            self._std_weight_velocity * x[3],
            1e-5,
            self._std_weight_velocity * x[3]]
        
        Q = np.diag(np.square(np.r_[std_pos, std_vel]))                        # 噪声矩阵 Q
        
        
        x = self.F @ x                                       # x = Fx + Bu
        P = self.F @ P @ self.F.T + Q                        # P = FPF.T + Q
        
        return x, P
    

    def project(self, x, P):
        """
        将状态空间的变量映射到测量空间。

        参数
        ----------
        x : ndarray
            状态向量的均值 (8 dimensional array).
        P : ndarray
            状态变量之间的协方差 (8x8 dimensional).

        返回
        -------
        (ndarray, ndarray)
            返回映射后的均值和协方差矩阵。
        """
        
        # 测量的标准差
        std_r = [
            self._std_weight_position * x[3],
            self._std_weight_position * x[3],
            1e-1,
            self._std_weight_position * x[3]]
        R = np.diag(np.square(std_r))               # 测量协方差
        
        z0 = self.H @ x                             # 状态空间映射到测量空间
        S = self.H @ P @ self.H.T + R               # 系统的协方差(不确定性)
        
        return z0, S
        
        
        
        
        
        
        pass


    def update(self, x, P, measurement):
        '''
        运行卡尔曼滤波器的更新步骤

        参数
        ----------
        x : ndarray
            预测阶段之后的状态变量 (8 dimensional).
        P : ndarray
            预测之后的状态变量的协方差 (8x8 dimensional).
        measurement : ndarray
            4维度的传感器测量值 (x, y, a, h), 其中 (x, y)是中心点位置, a 是宽高比, h 表示边界框的高度。
        返回值
        -------
        (ndarray, ndarray)
            返回后验估计值以及其协方差矩阵
        '''
        z0, S = self.project(x, P)                  # 计算状态到测量的转换, 以及系统的不确定性 S = HPH.T + R
        y = measurement - z0                        # 测量值与先验估计的残差, y = z -Hx
        
        # K = PH.T @ S^-1
        # 因为这里存在求逆问题,所以采用cholesky分解
        # 采用求解方程组的格式: Ax=B -> x=(A^-1)B
        # 原始的 K = P @ H.T @ (S^-1)
        # K.T = (S^-1).T @ H @ P.T
        # S = HPH.T + R, 所以S是实对称正定矩阵, S.T = S
        # 所以 K.T = (S^-1) @ H @ P.T
        # K.T = (S^-1) @ (P @ H.T).T, 假设 A=S, x=K.T, B=HP.T, 则该等式表示为Ax=B -> x=(A^-1)B
        # K = ((S^-1) @ (P @ H.T).T).T
        # scipy的求解方式为
        chol_factor, lower = scipy.linalg.cho_factor(S, lower=True, check_finite=False)
        K = scipy.linalg.cho_solve((chol_factor, lower), np.dot(P, self.H.T).T, check_finite=False).T    # 卡尔曼增益
        
        x = x + K @ y                                           # 后验估计值
        P = P - np.linalg.multi_dot((K, S, K.T))                # K=PH.T(S^-1), 所以KSK.T=KHP.T=KHP
        # P = P - np.linalg.multi_dot((K, self.H, P))
        
        return x, P


    def gating_distance(self, mean, covariance, measurements, only_position=False):
        """
        计算状态分布和测量之间的门控距离(gating distance), 也就是所谓的马氏距离。
        这里计算的是某一个跟踪目标与所有检测之间的马氏距离

        一个合适的距离阈值可能包含在 `chi2inv95`。 如果`only_position` 是 False, 那么 chi-square 分布有 4 个自由度, 否则 2 个。

        参数
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        返回值
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        mean, covariance = self.project(mean, covariance)               # 将状态变量映射到测量空间, Hx, S
        
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]             # 只取边界框中心点(x,y)的均值和方差
            measurements = measurements[:, :2]                          # 所有测量的中心点
            
        # 欧式距离会被不同维度的量纲所影响，因此马氏距离定义为去掉量纲后的欧式距离（这里是去掉协方差矩阵）
        # 马氏距离的计算: D(x,y) = sqrt((x-y).T * (cov^-1) * (x-y))
        # 由于协方差矩阵的的逆难以计算,因此考虑采用Cholesky分解（协方差矩阵是实对称半正定的，分解后L为下三角矩阵）
        # Cholesky分解定义为 ∑=𝐿𝐿𝑇
        # 代入后的马氏距离为： 𝐷(𝑥,𝑦)=sqrt([𝐿−1(𝑥−𝑦)]𝑇[𝐿−1(𝑥−𝑦)])
        L = np.linalg.cholesky(covariance)                              # 协方差矩阵的巧克力分解,可以得到下三角矩阵
        d = measurements - mean                                         # 残差, 测量与估计之间的差值                
        
        # 接下来,求解 (L^-1)(x-y), 我们假设d = x-y, L(L^-1)d = d, 所以假设 A=L, x=(L^-1)d, B=d, 则 Ax=B -> x=(A^-1)B=(L^-1)d
        # 考虑到,我们的状态变量mean的维度是[1, n], 所以应该是[n, 1],为了表示方便,我们输入是[1, n], 因此这里计算的时候需要转置
        # 下面的公式是计算Ax=B, 且A表示三角矩阵
        z = scipy.linalg.solve_triangular(L, d.T, lower=True, check_finite=False, overwrite_b=True)
        maha_distance_squared = np.sum(z * z, axis=0)                   # 马氏距离, 还可以写为 sum(z.T @ z)
        
        return maha_distance_squared





































