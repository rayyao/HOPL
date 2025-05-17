
import os
import cv2
import sys
from os.path import join, isdir, abspath, dirname
import numpy as np
import argparse

from HOPL.lib.test.tracker.vipt import ViPTTrack
import HOPL.lib.test.parameter.vipt as rgbt_prompt_params
import multiprocessing
import torch
from HOPL.lib.train.dataset.depth_utils import get_HSItest_frame
import time
from HOPL.lib.test.tracker.ostrack import OSTrack
import math
from HOPL.lib.models.hopl import build_viptrack
from HOPL.lib.test.tracker.basetracker import BaseTracker
import torch
from HOPL.lib.test.tracker.vis_utils import gen_visualization
from HOPL.lib.test.utils.hann import hann2d
from HOPL.lib.train.data.processing_utils import sample_target
import cv2
import os
#import vot
from HOPL.lib.test.tracker.data_utils import PreprocessorMM
from HOPL.lib.utils.box_ops import clip_box
from HOPL.lib.utils.ce_utils import generate_mask_cond
import numpy as np
#from HOPL.lib.SAM import sam
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
prj = join(dirname(__file__), '..')


class Frame:
    def __init__(self, id, diff):
        self.id = id
        self.diff = diff

    def __lt__(self, other):
        return self.id < other.id

    def __gt__(self, other):
        return other.__lt__(self)

    def __eq__(self, other):
        return self.id == other.id and self.id == other.id

    def __ne__(self, other):
        return not self.__eq__(other)


def rel_change(prev, curr):
    return abs(curr - prev) / prev if prev != 0 else 0


def smooth(x, window_len=13):
    s = np.r_[2 * x[0] - x[window_len:1:-1],
              x, 2 * x[-1] - x[-1:-window_len:-1]]
    w = np.hanning(window_len)
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len - 1:-window_len + 1]


def extract_keyframe_difference(frame1, frame2, len_window=50, use_thresh=True, thresh=0.87):
    luv1 = frame1
    luv2 = frame2
    diff = cv2.absdiff(luv1, luv2)
    diff_sum = np.sum(diff)
    diff_sum_mean = diff_sum / (diff.shape[0] * diff.shape[1])
    return diff_sum_mean

if prj not in sys.path:
    sys.path.append(prj)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def map_box_back(state, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = state[0] + 0.5 * state[2], state[1] + 0.5 * state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * 256/ resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
def show_box1(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='red', facecolor=(0, 0, 0, 0), lw=2))

def genConfig(seq_path, set_type):
    name = os.path.basename(seq_path)
    # RGB_img_list = sorted(
    #     ['/home/zl/HSI2023/test/HSI-VIS-FalseColor/'+name+ '/' + p for p in os.listdir('/home/zl/HSI2023/test/HSI-VIS-FalseColor/'+name +'/' ) if p.endswith(".jpg")])
    HSI_img_list = sorted(
        [seq_path + '/' + p for p in os.listdir(seq_path+'/') if p.endswith(".png")])
    #RGB_txt_path = os.path.join(seq_path  + '/groundtruth_rect.txt')
    HSI_txt_path = os.path.join(seq_path  + '/groundtruth_rect.txt')
    # with open(RGB_txt_path, 'r') as file:
    #     data = file.readlines()
    #     RGB_data = [list(map(float, line.strip().split('\t'))) for line in data]
    with open(HSI_txt_path, 'r') as file:
        data = file.readlines()
        HSI_data = [list(map(float, line.strip().split('\t'))) for line in data]
    #RGB_gt = np.array(RGB_data)
    HSI_gt = np.array(HSI_data)
    return  HSI_img_list, HSI_gt

def run_sequence(seq_name, seq_home, dataset_name, yaml_name, num_gpu=1, epoch=300, debug=0, script_name='prompt'):
    if 'VTUAV' in dataset_name:
        seq_txt = seq_name.split('/')[1]
    else:
        seq_txt = seq_name
    save_name = '{}'.format(yaml_name)
    save_path = f'/viptnew/results/{dataset_name}/' + save_name + '/' + seq_txt + '.txt'
    save_folder = f'/viptnew/results/{dataset_name}/' + save_name
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    if os.path.exists(save_path):
        print(f'-1 {seq_name}')
        return
    try:
        worker_name = multiprocessing.current_process().name
        worker_id = int(worker_name[worker_name.find('-') + 1:]) - 1
        gpu_id = worker_id % num_gpu
        torch.cuda.set_device(2)
    except:
        pass
    if script_name == 'hopl':
        params = rgbt_prompt_params.parameters(yaml_name, epoch)
        mmtrack = ViPTTrack(params)
        tracker = ViPT_RGBHSI(tracker=mmtrack)
    seq_path = seq_home + '/' + seq_name
    print('——————————Process sequence: ' + seq_name + '——————————————')
    HSI_img_list, HSI_gt = genConfig(seq_path, dataset_name)
    if len(HSI_img_list) == len(HSI_gt):
        result = np.zeros_like(HSI_gt)
    else:
        result = np.zeros((len(HSI_img_list), 4), dtype=HSI_gt.dtype)
    result[0] = np.copy(HSI_gt[0])
    toc = 0
    for frame_idx, HSI_path in enumerate(HSI_img_list):
        tic = cv2.getTickCount()
        if frame_idx == 0:
            # initialization
            last_path=f'/HSI2023/test/ALL/' + seq_txt + '/' + '{:04}.png'.format(
                1)
            image= get_HSItest_frame(HSI_path, last_path, frame_idx,HSI_gt[0],seq_txt,
                                      dtype=getattr(params.cfg.DATA, 'XTYPE', 'rgbrgb'))
            tracker.initialize(image, HSI_gt[0].tolist())
        elif frame_idx > 0:
            last_path = f'/HSI2023/test/ALL/' + seq_txt  + '/' + '{:04}.png'.format(
                frame_idx)
            bbox=result[frame_idx-1]
            image = get_HSItest_frame( HSI_path, last_path, frame_idx,bbox,seq_txt,
                                      dtype=getattr(params.cfg.DATA, 'XTYPE', 'rgbrgb'))
            region, confidence = tracker.track(image)
            result[frame_idx] = np.array(region)
        toc += cv2.getTickCount() - tic
    if not debug:
        np.savetxt(save_path, result)
    print('{} , fps:{}'.format(seq_name, frame_idx / toc))


class ViPT_RGBHSI(object):
    def __init__(self, tracker):
        self.tracker = tracker

    def initialize(self, image, region):
        self.H, self.W, _ = image.shape
        gt_bbox_np = np.array(region).astype(np.float32)

        init_info = {'init_bbox': list(gt_bbox_np)}  # input must be (x,y,w,h)
        self.tracker.initialize(image, init_info)

    def track(self, img_RGB):
        '''TRACK'''
        outputs = self.tracker.track(img_RGB)
        pred_bbox = outputs['target_bbox']
        pred_score = outputs['best_score']
        return pred_bbox, pred_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run tracker on RGBHSI dataset.')
    parser.add_argument('--script_name', type=str, default='hopl',
                        help='Name of tracking method(ostrack, prompt, ftuning).')
    parser.add_argument('--yaml_name', type=str, default='shaw_rgbt',
                        help='Name of tracking method.')  # vitb_256_mae_ce_32x4_ep300 vitb_256_mae_ce_32x4_ep60_prompt_i32v21_onlylasher_rgbt
    parser.add_argument('--dataset_name', type=str, default='HOPL_test',
                        help='Name of dataset (GTOT,RGBT234,LasHeR,VTUAVST,VTUAVLT).')
    parser.add_argument('--threads', default=2, type=int, help='Number of threads')
    parser.add_argument('--num_gpus', default=3, type=int, help='Number of gpus')
    parser.add_argument('--epoch', default=60, type=int, help='epochs of ckpt')
    parser.add_argument('--mode', default='sequential', type=str, help='sequential or parallel')
    parser.add_argument('--debug', default=0, type=int, help='to vis tracking results')
    parser.add_argument('--video', default='', type=str, help='specific video name')
    args = parser.parse_args()
    yaml_name = args.yaml_name
    dataset_name = args.dataset_name
    seq_list = None
    seq_home = '/HSI2023/test/ALL/'
    seq_list = [f for f in os.listdir(seq_home) if isdir(join(seq_home, f))]
    seq_list.sort()
    start = time.time()
    if args.mode == 'parallel':
        sequence_list = [
            (s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch, args.debug, args.script_name) for s
            in seq_list]
        multiprocessing.set_start_method('spawn', force=True)
        with multiprocessing.Pool(processes=args.threads) as pool:
            pool.starmap(run_sequence, sequence_list)
    else:
        seq_list = [args.video] if args.video != '' else seq_list
        sequence_list = [
            (s, seq_home, dataset_name, args.yaml_name, args.num_gpus, args.epoch, args.debug, args.script_name) for s
            in seq_list]
        for seqlist in sequence_list:
            run_sequence(*seqlist)
    print(f"Totally cost {time.time() - start} seconds!")
