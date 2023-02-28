#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   deep-sort-app.py
@Time    :   2023/02/27 15:14:15
@Author  :   JianBo Sun 
@Version :   1.0
@License :   (C)Copyright 2021-2022, ai-uav-cyy-cmii
@Desc    :   None
'''

import numpy as np
import zmq
import cv2


from detection import Detection
from nn_matching import NearestNeighborDistanceMetric
from tracker import Tracker
from track import TrackState
import preprocess




def non_maximum_suppression(detections):
    
    # detections是一个二维数组, [tl x, tl y, w, h, conf], 维度是 [n, 5]
    # 转置后解包,类似于yolov5的操作
    left, top, width, height = detections[:, :4].T
    right  = left + width - 1                                       # 计算检测框右下角坐标
    bottom = top + height - 1
    detections[:, 2] = right
    detections[:, 3] = bottom
    detections       = preprocess.nms(detections, 0.25, 4)                      # 非极大值抑制, NMS操作
    # 至此, detections内容变为 [tl x, tly, r, b, conf], 维度是 [n, 5]
    detections[:, 2:4] = detections[:, 2:4] - detections[:, :2] + 1            # 计算宽高
    
    return detections



if __name__ == "__main__":
    

    image_dir      = "/workspace/DeepSort/PETS09-S2L1/img1"                                         # 图像的目录
    detection_file = "/workspace/DeepSort/PETS09-S2L1/det/det.txt"                                  # 目标检测结果文件路径
    annotations = preprocess.load_detection_annotations(detection_file)         # 加载目标检测结果

    # 获取图像ID,并去重. 目标检测结果的第0个位置表示图像的ID
    image_ids = np.unique(annotations[:, 0]).astype(int)

    metric = NearestNeighborDistanceMetric(metric="cosine", matching_threshold=0.5, budget=100)
    tracker = Tracker(metric)
    random_color = (np.random.rand(32, 3) * 255).astype(int)
    zmq_server = preprocess.ZmqShow(port=12345)

    for img_id in image_ids:
        
        # 获取第i张图像的检测结果
        # detections.shape = n x 10
        # 格式定义为：
        # https://motchallenge.net/data/MOT15/
        # https://github.com/dendorferpatrick/MOTChallengeEvalKit
        # <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        detections = annotations[annotations[:, 0] == img_id]           # 维度是 [n, 10]
        detections = detections[:, [2, 3, 4, 5, 6]]                     # 只保留[tl x, tl y, w, h, conf], 维度是 [n, 5]
        detections = non_maximum_suppression(detections)                # 返回值仍然是[tl x, tl y, w, h, conf], 维度是 [n, 5]
        

        bboxes, confidences = detections[:, :4], detections[:, 4:]
        detections = []
        for tlwh, conf in zip(bboxes, confidences):
            detections.append(Detection(tlwh, conf, feature=None))
        
        tracker.predict()
        tracker.update(detections)
        
        image_file = f"{image_dir}/{img_id:06d}.jpg"
        image = cv2.imread(image_file)
        cv2.putText(image, f"{img_id}", (10, 50), 0, 2, (0, 255, 0), 2, 16, False)
        
        for track in tracker.tracks:
            x, y, a, h = [item for item in track.mean[:4]]           # xyah
            w = a * h
            left, top = x - w/2, y - h/2
            right, bottom = x + w/2, y + h/2
            
            location = [int(left), int(top), int(right), int(bottom)]
            
            track_color = random_color[track.track_id % random_color.shape[0]]
            if track.state == TrackState.Confirmed and track.time_since_update == 0:
                preprocess.draw_box(image, location[:2], location[2:4], color=track_color, text=f"{track.track_id}", trace=None)
        

        img_data = cv2.imencode(".jpg", image)[1].tobytes()
        
        zmq_server.send(img_data)
        print(f"Process: {img_id}")
        


    




































