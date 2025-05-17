from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse
import sys
env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
import numpy as np
import pandas as pd


def region_to_bbox(region, center=True):
    n = len(region)
    assert n == 4 or n == 8, ('GT region format is invalid, should have 4 or 8 entries.')

    if n == 4:
        return _rect(region, center)
    else:
        return _poly(region, center)


# we assume the grountruth bounding boxes are saved with 0-indexing
def _rect(region, center):
    if center:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]
        cx = x + w / 2
        cy = y + h / 2
        return np.array([cx, cy, w, h])
    else:
        # region[0] -= 1
        # region[1] -= 1
        return region


def _poly(region, center):
    cx = np.mean(region[::2])
    cy = np.mean(region[1::2])
    x1 = np.min(region[::2])
    x2 = np.max(region[::2])
    y1 = np.min(region[1::2])
    y2 = np.max(region[1::2])
    A1 = np.linalg.norm(region[0:2] - region[2:4]) * np.linalg.norm(region[2:4] - region[4:6])
    A2 = (x2 - x1) * (y2 - y1)
    s = np.sqrt(A1 / A2)
    w = s * (x2 - x1) + 1
    h = s * (y2 - y1) + 1

    if center:
        return np.array([cx, cy, w, h])
    else:
        return np.array([cx - w / 2, cy - h / 2, w, h])

def _compute_distance(boxA, boxB):
    a = np.array((boxA[0]+boxA[2]/2, boxA[1]+boxA[3]/2))
    b = np.array((boxB[0]+boxB[2]/2, boxB[1]+boxB[3]/2))
    dist = np.linalg.norm(a - b)

    assert dist >= 0
    assert dist != float('Inf')

    return dist

def overlap_ratio(rect1, rect2):
    '''Compute overlap ratio between two rects
    Args
        rect:2d array of N x [x,y,w,h]
    Return:
        iou
    '''
    # if rect1.ndim==1:
    #     rect1 = rect1[np.newaxis, :]
    # if rect2.ndim==1:
    #     rect2 = rect2[np.newaxis, :]
    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = intersect / union
    iou = np.maximum(np.minimum(1, iou), 0)
    return iou

def success_overlap(gt_bb, result_bb, n_frame):
    thresholds_overlap = np.arange(0.02, 1.02, 0.02)
    success = np.zeros(len(thresholds_overlap))
    iou = np.ones(len(gt_bb)) * (-1)
    mask = np.sum(gt_bb > 0, axis=1) == 4
    iou[mask] = overlap_ratio(gt_bb[mask], result_bb[mask])
    for i in range(len(thresholds_overlap)):
        success[i] = np.sum(iou >= thresholds_overlap[i]) / float(n_frame)
    return success
def compile_results(gt, bboxes):
    l = np.size(bboxes, 0)
    gt4 = np.zeros((l, 4))
    new_distances = np.zeros(l)
    n_thresholds = 50
    ops = np.zeros(n_thresholds)
    distance_thresholds = np.linspace(1,50,50)
    dp_20 = np.zeros(50)
    precision=dp_20
    for i in range(l):
        gt4[i, :] = region_to_bbox(gt[i, :], center=False)
        new_distances[i] = _compute_distance(bboxes[i, :], gt4[i, :])
    for i in range(50):
        precision[i] =np.float(sum(new_distances < distance_thresholds[i])) / np.size(new_distances)
    dp_20 = precision[19]
    average_center_location_error =new_distances.mean()
    # what's the percentage of frame in which center displacement is inferior to given threshold? (OTB metric)

    success=success_overlap(gt4, bboxes, l)
    # integrate over the thresholds
    auc = np.mean(success)
    return precision,success,average_center_location_error, auc,dp_20

if __name__ == '__main__':
    folder_path='/home/zl/HSI2023/test/HSI-RedNIR/'
    result_path='/home/zl/HOT2023result/mht/RedNIR/'
    precisions = []
    successes = []
    center_location_errors = []
    aucs = []
    dp_20s = []

    dataset_names=sorted(next(os.walk(folder_path))[1])
    for dataset_name in dataset_names:
        dataset_folder_path=os.path.join(folder_path,dataset_name)
        result_folder_path = result_path
        gt_path=os.path.join(dataset_folder_path,"groundtruth_rect.txt")
        gt=np.loadtxt(gt_path)
        bboxes_path=os.path.join(result_folder_path,dataset_name+".txt")
        bboxes=np.loadtxt(bboxes_path)
        precision, success, average_center_location_error, auc, dp_20 = compile_results(gt, bboxes)
        # 将评估结果存储到列表中
        precisions.append(precision)
        successes.append(success)
        center_location_errors.append(average_center_location_error)
        aucs.append(auc)
        dp_20s.append(dp_20)

    # 计算综合的评估结果，取各个数据集结果的平均值
    avg_precision = np.mean(precisions, axis=0)
    avg_success = np.mean(successes, axis=0)
    avg_center_location_error = np.mean(center_location_errors)
    avg_auc = np.mean(aucs)
    avg_dp_20 = np.mean(dp_20s)
    m=0



