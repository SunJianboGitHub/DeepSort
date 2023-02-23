#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   preprocess.py
@Time    :   2023/02/22 10:15:56
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   目标跟踪之前的预处理,主要是目标检测的后处理
'''

import cv2
import zmq
import numpy as np


class ZmqShow:
    def __init__(self, port=12345):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")

    def send(self, img_data):
        message = self.socket.recv()
        print(f"Wait message, image_data = {len(img_data) / 1024:.2f} KB")
        print("message = ", message)
        self.socket.send(img_data)



# 根据文件名称,加载目标检测的结果
def load_detection_annotations(file):
    with open(file, "r") as f:
        lines = f.readlines()                                       # 读取所有行
        lines = [line.strip().split(",") for line in lines]         # 处理每一行,删除尾部换行符以及分割为列表
        annotations = np.array(lines, dtype=float)                  # 转换成浮点型

        return annotations


def iou(a, b):
    ax, ay, ar, ab = a
    bx, by, br, bb = b

    cross_x = max(ax, bx)
    cross_y = max(ay, by)
    cross_r = min(ar, br)
    cross_b = min(ab, bb)
    cross_w = max(0, (cross_r - cross_x) + 1)
    cross_h = max(0, (cross_b - cross_y) + 1)
    cross_area = cross_w * cross_h
    union = (ar - ax + 1) * (ab - ay + 1) + (br - bx + 1) * (bb - by + 1) - cross_area
    return cross_area / union

# 检测结果的后处理,非极大值抑制
def nms(detectiones, threshold, confidence_index=-1):

    detectiones = sorted(detectiones, key=lambda x: x[confidence_index], reverse=True)
    # detectiones = [detect for detect in detectiones if detect[-1] > 0.5]
    flags = [True] * len(detectiones)
    keep = []
    for i in range(len(detectiones)):
        if not flags[i]: continue
        keep.append(detectiones[i])

        for j in range(i+1, len(detectiones)):
            if iou(detectiones[i][:4], detectiones[j][:4]) > threshold:
                flags[j] = False
    return np.vstack(keep)



def draw_box(image, p1, p2, color=(0, 255, 0), text=None, trace=None):
    
    left, top = p1
    right, bottom = p2
    b, g, r = color
    color = int(b), int(g), int(r)
    cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), color, 3)

    if text is not None:
        cv2.putText(image, text, (int(left), int(top) - 10), 0, 1, color, 1, 16)

    if trace is not None:
        for detect in trace:
            x, y, r, b = detect[:4]
            cx, cy = int((x + r) * 0.5), int((y + b) * 0.5)
            cv2.circle(image, (cx, cy), radius=2, color=color, thickness=2, lineType=cv2.LINE_AA)



# 将检测结果转换为[cx, cy, w_h_ratio, height]，便于目标跟踪
def detection_to_xyah(detect):
    # detect[x, y, r, b] -> [cx, cy, w_h_ratio, height]
    left, top, right, bottom = detect[:4]
    center_x = (left + right) * 0.5
    center_y = (top + bottom) * 0.5
    width    = (right - left) + 1
    height   = (bottom - top) + 1
    w_h_ratio = width / height

    return np.array([center_x, center_y, w_h_ratio, height])


















