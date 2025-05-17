#
# import cv2
# import numpy as np
# from PIL import Image
# import math
# import random
# import torch
# import numpy as np
# import os
# import sys
# import time
# import argparse
# from PIL import Image
# import matplotlib.pyplot as plt
# import torch
# import torch.utils.data as data
# import torch.optim as optim
# import os
# import json
# import numpy as np
#
# order=None
# def gen_config(args, videoname):
#
#     if args.seq != '':
#         # generate config from a sequence name
#
#         # seq_home = 'datasets/OTB'
#         seq_home = args.seq
#         result_home = args.savepath
#
#         seq_name = videoname
#         img_dir = os.path.join(seq_home, seq_name, 'HSI')
#         #gt_path = os.path.join(seq_home, seq_name,'groundtruth_rect.txt')
#
#         img_list = os.listdir(img_dir)
#         img_list.sort()
#         img_list = [os.path.join(img_dir, x) for x in img_list]
#
#
#         result_dir = result_home # os.path.join(result_home, seq_name)
#         if not os.path.exists(result_dir):
#             os.makedirs(result_dir)
#
#     return img_list
#
# def X2Cube(img):
#
#     B = [4, 4]
#     skip = [4, 4]
#     # Parameters
#     M, N = img.shape
#     col_extent = N - B[1] + 1
#     row_extent = M - B[0] + 1
#
#     # Get Starting block indices
#     start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])
#
#     # Generate Depth indeces
#     didx = M * N * np.arange(1)
#     start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))
#
#     # Get offsetted indices across the height and width of input array
#     offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
#
#     # Get all actual indices & index into input array for final output
#     out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
#     out = np.transpose(out)
#     img = out.reshape(M//4, N//4, 16)
#     img = img / img.max() * 255 #  归一化
#     img.astype('uint8')
#     return img
#
#
# def Select(hsi_img):
#     w=[]
#     for kk in hsi_img:
#      a=kk.reshape(1,-1)
#      sum1=0
#      l=np.sum(a)
#      k=a.size
#      pj=l/k
#      for m in range(a.size):
#          b=(a[0,m]-pj)**2
#          sum1=sum1+b
#          d=(sum1/(a.size))**0.5
#      #d=(d,KK)
#      w.append(d)
#     w1=np.array(w)
#     w1=np.argsort(-w1)
#     orderW = w1[0:15]
#     return orderW
#
#
# def HSI2RGB(sample,order):
#     ordersample0 = sample[order[0]]
#     ordersample1 = sample[order[1]]
#     ordersample2= sample[order[2]]
#     com1 = np.array([ordersample0,ordersample1,ordersample2])
#     ordersample3 = sample[order[3]]
#     ordersample4 = sample[order[4]]
#     ordersample5 = sample[order[5]]
#     com2 = np.array([ordersample3, ordersample4, ordersample5])
#     return com1,com2
#
# # def HSI2RGB(sample,order):
# #     ordersample0 = sample[order[0]]
# #     ordersample1 = sample[order[1]]
# #     ordersample2= sample[order[2]]
# #     com1 = np.array([ordersample0,ordersample1,ordersample2])
# #
# #     return com1
#
#
# def entropy(hsi_img):
#     band_entropies = []
#     h, w, c = hsi_img.shape
#     for band in range(16):
#         current_band = hsi_img[:, :, band]
#         pixel_counts = np.histogram(current_band, bins=np.arange(256))[0]
#         pixel_probabilities = pixel_counts / np.sum(pixel_counts)
#         entropy = -np.sum(pixel_probabilities * np.log2(pixel_probabilities + 1e-10))
#         band_entropies.append(entropy)
#     return band_entropies
#
#
# def normalize(data):
#     min_data = min(data)
#     max_data = max(data)
#     normalize_data = [(x - min_data) / (max_data - min_data) for x in data]
#     return normalize_data
#
#
# def distance(coor_list):
#     distance_matrix = np.zeros((16, 16))
#     for i in range(16):
#         for j in range(16):
#             x1, y1 = coor_list[i]
#             x2, y2 = coor_list[j]
#             distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#             distance_matrix[i, j] = distance
#     return distance_matrix
#
#
# def distance_y(coor_list, a):
#     distance_matrix = np.zeros((16, 16))
#     for i in range(16):
#         for j in range(16):
#             x1, y1 = coor_list[i]
#             x2, y2 = coor_list[j]
#             distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
#             if y1 > y2 and i not in a:
#                 distance = 0
#             distance_matrix[i, j] = distance
#     return distance_matrix
#
#
# def y_coor(coor, a):
#     result_list = []
#     for i, row in enumerate(coor):
#         if i in a:
#             max_value = max(row)
#             result_list.append(max_value)
#         else:
#
#             non_zero_min = min(value for value in row if value != 0)
#             result_list.append(non_zero_min)
#     return result_list
#
#
# # def bandselect(hsi_img, i):
# #     hsi_img = np.array(hsi_img)
# #     global order,order2
# #     hsi_img = X2Cube(hsi_img)
# #     sample = hsi_img.transpose(2, 0, 1)
# #     if i == 0:
# #         hsi_entropy = entropy(sample)
# #         hsi_entropy = normalize(hsi_entropy)
# #         coordinates = [(index, value) for index, value in enumerate(hsi_entropy)]
# #         distance_matrix = distance(coordinates)
# #         max_value = np.max(distance_matrix)
# #         count_list = []
# #         for row in distance_matrix:
# #             count = np.sum(row < max_value / 3)
# #             count = count - 1
# #             count_list.append(count)
# #         max_count = max(count_list)
# #         max_count_list = [index for index, value in enumerate(count_list) if value == max_count]
# #         count_coordinates = [(index, value) for index, value in enumerate(count_list)]
# #         cont_distancematrix = distance_y(count_coordinates, max_count_list)
# #         y = y_coor(cont_distancematrix, max_count_list)
# #         order = sorted(range(len(y)), key=lambda i: y[i], reverse=True)
# #         order2 = sorted(range(len(hsi_entropy)), key=lambda i: hsi_entropy[i], reverse=True)
# #     rgb_image1= HSI2RGB(sample, order)
# #     #rgb_image2 = HSI2RGB(sample, order2)
# #     rgb_image1 = rgb_image1.transpose(1, 2, 0)
# #     #rgb_image2 = rgb_image2.transpose(1, 2, 0)
# #     #img = np.concatenate((rgb_image1, rgb_image2), axis=2)
# #     #return img
# #     return rgb_image1
#
#
# def bandselect(hsi_img, name):
#     hsi_img = np.array(hsi_img)
#     hsi_img = X2Cube(hsi_img)
#     sample = hsi_img.transpose(2, 0, 1)
#     if name=="automobile":
#         order=[5, 6, 7, 15, 14, 13]
#     if name=="automobile2":
#         order=[5, 6, 7,15, 14, 6 ]
#     if name=="automobile3":
#         order=[9, 3, 15, 15, 14, 13]
#     if name=="automobile4":
#         order=[5, 8, 2, 15, 14, 6]
#     if name=="automobile5":
#         order=[9, 3, 15, 15, 14, 6]
#     if name=="automobile6":
#         order=[5, 6, 7, 0, 1, 6]
#     if name=="automobile7":
#         order=[11, 5, 10, 15, 14, 9]
#     if name=="automobile8":
#         order=[10, 0, 1, 14, 13, 15]
#     if name=="automobile9":
#         order=[4, 11, 5, 15, 13, 14]
#     if name=="automobile10":
#         order=[10, 12, 0, 13, 14, 15]
#     if name=="automobile11":
#         order=[10, 12, 0, 13, 12, 14]
#     if name=="automobile12":
#         order=[6, 0, 12, 13, 12, 14]
#     if name=="automobile13":
#         order=[6, 0, 12, 13, 12, 14]
#     if name=="automobile14":
#         order=[6, 7, 0, 9, 11, 8]
#     if name=="basketball":
#         order=[10, 6, 0, 13, 14, 12]
#     if name=="board":
#         order=[5, 6, 7, 15, 14, 13]
#     if name=="bus":
#         order=[6, 0, 12, 0, 1, 9]
#     if name=="bus2":
#         order=[5, 10, 9, 12, 9, 13]
#     if name=="car1":
#         order=[10, 5, 11, 12, 8, 13]
#     if name=="car2":
#         order=[6, 0, 3, 13, 10, 12]
#     if name=="car3":
#          order=[4,9,8, 1, 0, 2]
#     if name =="car4":
#         order = [10, 0, 1, 15, 14, 13]
#     if name =="car5":
#         order =[6, 0, 3,2, 0, 8]
#     if name =="car6":
#         order =[9, 3, 15, 14, 15, 13]
#     if name =="car7":
#         order =[5, 10, 9, 11, 14, 13]
#     if name =="car8":
#         order =[5, 10, 9, 11, 14, 13]
#     if name =="car9":
#         order =[10, 5, 0,15, 14, 13]
#     if name =="car10":
#         order =[10, 9, 3, 15, 14, 13]
#     if name =="kangaroo":
#         order =[11, 5, 6,14, 11, 6]
#     if name =="pedestrian":
#         order =[4, 11, 5,11, 10, 5]
#     if name =="pedestrian2":
#         order =[4, 11, 5, 2, 1, 15]
#     if name =="pedestrian3":
#         order =[4, 11, 5, 11, 10, 5]
#     if name =="pedestrian4":
#         order =[4, 11, 5, 11, 10, 5]
#     if name =="rider1":
#         order =[10, 0, 1,13, 12, 14]
#     if name =="rider2":
#         order =[4, 11, 5, 11, 9, 12]
#     if name =="rider3":
#         order =[10, 9, 8, 1, 0, 12]
#     if name =="rider4":
#         order =[7, 1, 0, 15, 14, 13]
#     if name =="taxi":
#         order =[5, 9, 8,11, 12, 10]
#     if name =="toy":
#         order =[5, 10, 6,1, 0, 2]
#     if name =="toy2":
#         order =[5, 6, 9,15, 14, 13]
#     rgb_image1,rgb_image2= HSI2RGB(sample, order)
#     rgb_image1 = rgb_image1.transpose(1, 2, 0)
#     rgb_image2 = rgb_image2.transpose(1, 2, 0)
#     img = np.concatenate((rgb_image1, rgb_image2), axis=2)
#     return img

import cv2
import numpy as np
from PIL import Image
import math
import random
import torch
import numpy as np
import os
import sys
import time
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torch.optim as optim
import os
import json
import numpy as np

order=None
def gen_config(args, videoname):

    if args.seq != '':
        # generate config from a sequence name

        # seq_home = 'datasets/OTB'
        seq_home = args.seq
        result_home = args.savepath

        seq_name = videoname
        img_dir = os.path.join(seq_home, seq_name, 'HSI')
        #gt_path = os.path.join(seq_home, seq_name,'groundtruth_rect.txt')

        img_list = os.listdir(img_dir)
        img_list.sort()
        img_list = [os.path.join(img_dir, x) for x in img_list]


        result_dir = result_home # os.path.join(result_home, seq_name)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

    return img_list

def X2Cube(img):

    B = [4, 4]
    skip = [4, 4]
    # Parameters
    M, N = img.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1

    # Get Starting block indices
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])

    # Generate Depth indeces
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))

    # Get offsetted indices across the height and width of input array
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)

    # Get all actual indices & index into input array for final output
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    out = np.transpose(out)
    img = out.reshape(M//4, N//4, 16)
    img = img / img.max() * 255 #  归一化
    img.astype('uint8')
    return img


def Select(hsi_img):
    w=[]
    for kk in hsi_img:
     a=kk.reshape(1,-1)
     sum1=0
     l=np.sum(a)
     k=a.size
     pj=l/k
     for m in range(a.size):
         b=(a[0,m]-pj)**2
         sum1=sum1+b
         d=(sum1/(a.size))**0.5
     #d=(d,KK)
     w.append(d)
    w1=np.array(w)
    w1=np.argsort(-w1)
    orderW = w1[0:15]
    return orderW

def HSI2RGB(sample,order):
    ordersample0 = sample[order[0]]
    ordersample1 = sample[order[1]]
    ordersample2= sample[order[2]]
    com1 = np.array([ordersample0,ordersample1,ordersample2])
    ordersample3 = sample[order[3]]
    ordersample4 = sample[order[4]]
    ordersample5 = sample[order[5]]
    com2 = np.array([ordersample3, ordersample4, ordersample5])
    return com1,com2
# def HSI2RGB(sample,order):
#     ordersample0 = sample[order[0]]
#     ordersample1 = sample[order[1]]
#     ordersample2= sample[order[2]]
#     com1 = np.array([ordersample0,ordersample1,ordersample2])
#
#     return com1

def entropy(hsi_img):
    band_entropies = []
    h, w, c = hsi_img.shape
    for band in range(16):
        current_band = hsi_img[:, :, band]
        pixel_counts = np.histogram(current_band, bins=np.arange(256))[0]
        pixel_probabilities = pixel_counts / np.sum(pixel_counts)
        entropy = -np.sum(pixel_probabilities * np.log2(pixel_probabilities + 1e-10))
        band_entropies.append(entropy)
    return band_entropies


def normalize(data):
    min_data = min(data)
    max_data = max(data)
    normalize_data = [(x - min_data) / (max_data - min_data) for x in data]
    return normalize_data


def distance(coor_list):
    distance_matrix = np.zeros((16, 16))
    for i in range(16):
        for j in range(16):
            x1, y1 = coor_list[i]
            x2, y2 = coor_list[j]
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            distance_matrix[i, j] = distance
    return distance_matrix


def distance_y(coor_list, a):
    distance_matrix = np.zeros((16, 16))
    for i in range(16):
        for j in range(16):
            x1, y1 = coor_list[i]
            x2, y2 = coor_list[j]
            distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if y1 > y2 and i not in a:
                distance = 0
            distance_matrix[i, j] = distance
    return distance_matrix


def y_coor(coor, a):
    result_list = []
    for i, row in enumerate(coor):
        if i in a:
            max_value = max(row)
            result_list.append(max_value)
        else:

            non_zero_min = min(value for value in row if value != 0)
            result_list.append(non_zero_min)
    return result_list

#
# def bandselecttest(hsi_img, i):
#     hsi_img = np.array(hsi_img)
#     global order,order2
#     hsi_img = X2Cube(hsi_img)
#     sample = hsi_img.transpose(2, 0, 1)
#     if i == 0:
#         hsi_entropy = entropy(sample)
#         hsi_entropy = normalize(hsi_entropy)
#         coordinates = [(index, value) for index, value in enumerate(hsi_entropy)]
#         distance_matrix = distance(coordinates)
#         max_value = np.max(distance_matrix)
#         count_list = []
#         for row in distance_matrix:
#             count = np.sum(row < max_value / 3)
#             count = count - 1
#             count_list.append(count)
#         max_count = max(count_list)
#         max_count_list = [index for index, value in enumerate(count_list) if value == max_count]
#         count_coordinates = [(index, value) for index, value in enumerate(count_list)]
#         cont_distancematrix = distance_y(count_coordinates, max_count_list)
#         y = y_coor(cont_distancematrix, max_count_list)
#         order = sorted(range(len(y)), key=lambda i: y[i], reverse=True)
#         order2 = sorted(range(len(hsi_entropy)), key=lambda i: hsi_entropy[i], reverse=True)
#     rgb_image1= HSI2RGB(sample, order)
#     rgb_image2 = HSI2RGB(sample, order2)
#     rgb_image1 = rgb_image1.transpose(1, 2, 0)
#     rgb_image2 = rgb_image2.transpose(1, 2, 0)
#     #img = np.concatenate((rgb_image1, rgb_image2), axis=2)
#     return rgb_image1
def Allx2cube(img, i, band_number):
    B = [i, i]
    skip = [i, i]
    M, N = img.shape
    col_extent = N - B[1] + 1
    row_extent = M - B[0] + 1
    start_idx = np.arange(B[0])[:, None] * N + np.arange(B[1])
    didx = M * N * np.arange(1)
    start_idx = (didx[:, None] + start_idx.ravel()).reshape((-1, B[0], B[1]))
    offset_idx = np.arange(row_extent)[:, None] * N + np.arange(col_extent)
    out = np.take(img, start_idx.ravel()[:, None] + offset_idx[::skip[0], ::skip[1]].ravel())
    out = np.transpose(out)
    if band_number==15:
        out=out[:,:15]
    img = out.reshape(M // i, N // i, band_number)
    img = img / img.max() * 255  # 归一化
    img.astype('uint8')
    return img

def bandselecttest(hsi_img, i,name):
    hsi_img = np.array(hsi_img)
    global order,order2

    if 'vis' in name:
        band_number=16
        nn=4
    elif 'rednir' in name:
        band_number = 15
        nn = 4
    else:
        band_number = 25
        nn=5
    hsi_img = Allx2cube(hsi_img, nn, band_number)
    sample = hsi_img.transpose(2, 0, 1)
    if i == 0:
        hsi_entropy = entropy(sample)
        hsi_entropy = normalize(hsi_entropy)
        coordinates = [(index, value) for index, value in enumerate(hsi_entropy)]
        distance_matrix = distance(coordinates)
        max_value = np.max(distance_matrix)
        count_list = []
        for row in distance_matrix:
            count = np.sum(row < max_value / 3)
            count = count - 1
            count_list.append(count)
        max_count = max(count_list)
        max_count_list = [index for index, value in enumerate(count_list) if value == max_count]
        count_coordinates = [(index, value) for index, value in enumerate(count_list)]
        cont_distancematrix = distance_y(count_coordinates, max_count_list)
        y = y_coor(cont_distancematrix, max_count_list)
        order = sorted(range(len(y)), key=lambda i: y[i], reverse=True)
    rgb_image1,rgb_image2= HSI2RGB(sample, order)
    rgb_image1 = rgb_image1.transpose(1, 2, 0)
    rgb_image2 = rgb_image2.transpose(1, 2, 0)
    img = np.concatenate((rgb_image1, rgb_image2), axis=2)

    return img

def bandselecttrain(hsi_img, name):
    hsi_img = np.array(hsi_img)
    if 'vis' in name:
        band_number=16
        nn=4
    elif 'rednir' in name:
        band_number = 15
        nn = 4
    else:
        band_number = 25
        nn=5
    # if 'vis'and'rednir'and'nir' not in name:
    #     band_number = 16
    #     nn = 4
    hsi_img = Allx2cube(hsi_img, nn, band_number)
    sample = hsi_img.transpose(2, 0, 1)
    # if name=="automobile":
    #     order=[5, 6, 7, 0, 1, 4, 12, 13, 14, 15, 2, 3, 8, 9, 10, 11]
    # if name=="automobile2":
    #     order=[5, 6, 7, 8, 14, 0, 1, 4, 15, 2, 3, 9, 10, 11, 12, 13]
    # if name=="automobile3":
    #     order=[9, 3, 15, 0, 1, 2, 7, 8, 14, 4, 5, 6, 10, 11, 12, 13]
    # if name=="automobile4":
    #     order=[5, 8, 2, 14, 3, 4, 6, 7, 13, 15, 0, 1, 9, 10, 11, 12]
    # if name=="automobile5":
    #     order=[9, 3, 15, 0, 1, 2, 12, 13, 14, 4, 5, 6, 7, 8, 10, 11]
    # if name=="automobile6":
    #     order=[5, 6, 7, 8, 14, 0, 1, 2, 3, 4, 12, 13, 15, 9, 10, 11]
    # if name=="automobile7":
    #     order=[11, 5, 10, 6, 9, 7, 8, 15, 0, 1, 2, 3, 4, 12, 13, 14]
    # if name=="automobile8":
    #     order=[10, 0, 1, 2, 3, 11, 12, 13, 4, 5, 6, 7, 8, 9, 14, 15]
    # if name=="automobile9":
    #     order=[4, 11, 5, 10, 6, 9, 7, 8, 2, 13, 0, 1, 3, 12, 14, 15]
    # if name=="automobile10":
    #     order=[10, 12, 0, 1, 2, 3, 4, 11, 13, 5, 6, 7, 8, 9, 14, 15]
    # if name=="automobile11":
    #     order=[10, 12, 0, 1, 2, 3, 4, 9, 11, 15, 5, 6, 7, 8, 13, 14]
    # if name=="automobile12":
    #     order=[6, 0, 12, 1, 2, 3, 13, 14, 15, 4, 5, 7, 8, 9, 10, 11]
    # if name=="automobile13":
    #     order=[10, 2, 13, 0, 1, 3, 11, 12, 4, 5, 6, 7, 8, 9, 14, 15]
    # if name=="automobile14":
    #     order=[6, 7, 0, 13, 3, 14, 15, 1, 2, 4, 5, 8, 9, 10, 11, 12]
    # if name=="ball&mirror7":
    #     order=[10, 9, 8, 7, 1, 0, 2, 3, 11, 12, 13, 14, 15, 4, 5, 6]
    # if name=="basketball":
    #     order=[10, 6, 0, 3, 11, 12, 13, 1, 2, 4, 5, 7, 8, 9, 14, 15]
    # if name=="board":
    #     order=[5, 6, 7, 13, 0, 1, 2, 3, 4, 12, 14, 15, 8, 9, 10, 11]
    # if name=="bus":
    #     order=[6, 0, 12, 1, 2, 5, 9, 13, 14, 15, 3, 4, 7, 8, 10, 11]
    # if name=="bus2":
    #     order=[5, 10, 9, 2, 3, 4, 11, 12, 15, 0, 1, 6, 7, 8, 13, 14]
    # if name=="bytheriver1":
    #     order=[10, 9, 8, 2, 0, 1, 3, 11, 12, 13, 14, 15, 4, 5, 6, 7]
    # if name=="car1":
    #     order=[10, 5, 11, 0, 1, 2, 3, 4, 14, 15, 6, 7, 8, 9, 12, 13]
    # if name=="car2":
    #     order=[6, 0, 3, 12, 13, 14, 15, 1, 2, 4, 5, 7, 8, 9, 10, 11]
    # if name=="car3":
    #      order=[4, 9, 8, 3, 14, 0, 1, 2, 15, 5, 6, 7, 10, 11, 12, 13]
    # if name =="car4":
    #     order = [10, 0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 5, 6, 7, 8, 9]
    # if name =="car5":
    #     order =[6, 0, 3, 12, 13, 14, 15, 1, 2, 4, 5, 7, 8, 9, 10, 11]
    # if name =="car6":
    #     order =[9, 3, 15, 0, 1, 2, 12, 13, 14, 4, 5, 6, 7, 8, 10, 11]
    # if name =="car7":
    #     order =[5, 10, 9, 2, 3, 4, 11, 12, 15, 0, 1, 6, 7, 8, 13, 14]
    # if name =="car8":
    #     order =[5, 10, 9, 2, 3, 4, 11, 12, 15, 0, 1, 6, 7, 8, 13, 14]
    # if name =="car9":
    #     order =[10, 5, 0, 1, 2, 3, 4, 6, 13, 14, 15, 7, 8, 9, 11, 12]
    # if name =="car10":
    #     order =[10, 9, 3, 0, 1, 2, 11, 12, 13, 14, 15, 4, 5, 6, 7, 8]
    # if name =="cards11":
    #     order =[10, 9, 6, 8, 7, 0, 1, 2, 3, 11, 12, 13, 14, 15, 4, 5]
    # if name =="cloth1":
    #     order =[5, 2, 3, 4, 6, 11, 12, 13, 14, 15, 0, 1, 7, 8, 9, 10]
    # if name =="dice1":
    #     order =[10, 9, 8, 7, 1, 0, 2, 3, 11, 12, 13, 14, 15, 4, 5, 6]
    # if name =="duck3":
    #     order =[10, 9, 6, 8, 7, 0, 1, 2, 3, 11, 12, 13, 14, 15, 4, 5]
    # if name == "glass2":
    #     order =[4, 11, 5, 10, 6, 9, 3, 12, 0, 1, 2, 13, 14, 15, 7, 8]
    # if name =="kangaroo":
    #     order =[11, 5, 6, 7, 8, 14, 2, 3, 4, 12, 13, 15, 0, 1, 9, 10]
    # if name =="officechair1":
    #     order =[10, 9, 6, 8, 7, 0, 1, 2, 3, 11, 12, 13, 14, 15, 4, 5]
    # if name =="officefan2":
    #     order =[10, 9, 8, 7, 1, 0, 2, 3, 11, 12, 13, 14, 15, 4, 5, 6]
    # if name =="partylights3":
    #     order =[10, 9, 8, 7, 1, 0, 2, 3, 11, 12, 13, 14, 15, 4, 5, 6]
    # if name =="pedestrian":
    #     order =[4, 11, 5, 10, 6, 9, 7, 8, 0, 1, 2, 3, 12, 13, 14, 15]
    # if name =="pedestrian2":
    #     order =[4, 11, 5, 10, 6, 9, 7, 8, 0, 1, 2, 3, 12, 13, 14, 15]
    # if name =="pedestrian3":
    #     order =[4, 11, 5, 10, 6, 9, 7, 8, 0, 1, 2, 3, 12, 13, 14, 15]
    # if name =="pedestrian4":
    #     order =[10, 0, 1, 2, 3, 11, 12, 13, 4, 5, 6, 7, 8, 9, 14, 15]
    # if name =="pool5":
    #     order =[9, 0, 4, 1, 2, 3, 5, 10, 11, 12, 15, 6, 7, 8, 13, 14]
    # if name =="rainystreet2":
    #     order =[11, 5, 10, 6, 9, 7, 8, 15, 0, 1, 2, 3, 4, 12, 13, 14]
    # if name =="rainystreet5":
    #     order =[10, 0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 5, 6, 7, 8, 9]
    # if name =="receipts3":
    #     order =[10, 9, 8, 7, 1, 0, 11, 14, 15, 2, 3, 4, 5, 6, 12, 13]
    # if name =="rider1":
    #     order =[10, 0, 1, 2, 3, 11, 12, 13, 4, 5, 6, 7, 8, 9, 14, 15]
    # if name =="rider2":
    #     order =[4, 11, 5, 10, 6, 9, 7, 8, 0, 1, 2, 3, 12, 13, 14, 15]
    # if name =="rider3":
    #     order =[10, 9, 8, 2, 11, 0, 1, 3, 14, 15, 4, 5, 6, 7, 12, 13]
    # if name =="rider4":
    #     order =[7, 1, 0, 12, 13, 14, 15, 2, 3, 4, 5, 6, 8, 9, 10, 11]
    # if name =="taxi":
    #     order =[5, 9, 8, 15, 2, 3, 4, 14, 0, 1, 6, 7, 10, 11, 12, 13]
    # if name =="toy":
    #     order =[5, 10, 6, 9, 7, 8, 0, 15, 1, 2, 3, 4, 11, 12, 13, 14]
    # if name =="toy2":
    #     order =[5, 6, 9, 8, 15, 0, 3, 4, 7, 12, 13, 14, 1, 2, 10, 11]
    # if name =="whitecup3":
    #     order =[10, 9, 6, 8, 7, 0, 1, 2, 3, 11, 12, 13, 14, 15, 4, 5]
    if name == "vis_automobile":
        order =[5, 6, 7, 0, 1, 4, 12, 13, 14, 15, 2, 3, 8, 9, 10, 11]
    if name == "vis_automobile12":
        order =[6, 0, 12, 1, 2, 3, 13, 14, 15, 4, 5, 7, 8, 9, 10, 11]
    if name == "vis_receipts3":
        order =[10, 9, 8, 7, 1, 0, 11, 14, 15, 2, 3, 4, 5, 6, 12, 13]
    if name == "vis_rainystreet5":
        order =[10, 0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 5, 6, 7, 8, 9]
    if name == "vis_car8":
        order =[10, 9, 8, 2, 0, 1, 3, 11, 12, 13, 14, 15, 4, 5, 6, 7]
    if name == "vis_pedestrian2":
        order =[4, 11, 5, 10, 6, 9, 7, 8, 0, 1, 2, 3, 12, 13, 14, 15]
    if name == "vis_automobile7":
        order =[11, 5, 10, 6, 9, 7, 8, 15, 0, 1, 2, 3, 4, 12, 13, 14]
    if name == "vis_bus2":
        order =[5, 10, 9, 2, 3, 4, 11, 12, 15, 0, 1, 6, 7, 8, 13, 14]
    if name == "vis_toy2":
        order =[5, 6, 9, 8, 15, 0, 3, 4, 7, 12, 13, 14, 1, 2, 10, 11]
    if name == "vis_pedestrian3":
        order =[4, 11, 5, 10, 6, 9, 7, 8, 0, 1, 2, 3, 12, 13, 14, 15]
    if name == "vis_automobile13":
        order =[10, 2, 13, 0, 1, 3, 11, 12, 4, 5, 6, 7, 8, 9, 14, 15]
    if name == "vis_car3":
        order =[4, 9, 8, 3, 14, 0, 1, 2, 15, 5, 6, 7, 10, 11, 12, 13]
    if name == "vis_pool5":
        order =[9, 0, 4, 1, 2, 3, 5, 10, 11, 12, 15, 6, 7, 8, 13, 14]
    if name == "vis_automobile14":
        order =[6, 7, 0, 13, 3, 14, 15, 1, 2, 4, 5, 8, 9, 10, 11, 12]
    if name == "vis_basketball":
        order =[10, 6, 0, 3, 11, 12, 13, 1, 2, 4, 5, 7, 8, 9, 14, 15]
    if name == "vis_car1":
        order =[10, 5, 11, 0, 1, 2, 3, 4, 14, 15, 6, 7, 8, 9, 12, 13]
    if name == "vis_car7":
        order =[5, 10, 9, 2, 3, 4, 11, 12, 15, 0, 1, 6, 7, 8, 13, 14]
    if name == "vis_dice1":
        order =[10, 9, 8, 7, 1, 0, 2, 3, 11, 12, 13, 14, 15, 4, 5, 6]
    if name == "vis_automobile4":
        order =[5, 8, 2, 14, 3, 4, 6, 7, 13, 15, 0, 1, 9, 10, 11, 12]
    if name == "vis_automobile11":
        order =[10, 12, 0, 1, 2, 3, 4, 9, 11, 15, 5, 6, 7, 8, 13, 14]
    if name == "vis_automobile5":
        order =[9, 3, 15, 0, 1, 2, 12, 13, 14, 4, 5, 6, 7, 8, 10, 11]
    if name == "vis_car2":
        order =[6, 0, 3, 12, 13, 14, 15, 1, 2, 4, 5, 7, 8, 9, 10, 11]
    if name == "vis_automobile8":
        order =[10, 0, 1, 2, 3, 11, 12, 13, 4, 5, 6, 7, 8, 9, 14, 15]
    if name == "vis_rainystreet2":
        order =[11, 5, 10, 6, 9, 7, 8, 15, 0, 1, 2, 3, 4, 12, 13, 14]
    if name == "vis_officefan2":
        order =[10, 9, 8, 7, 1, 0, 2, 3, 11, 12, 13, 14, 15, 4, 5, 6]
    if name == "vis_car10":
        order =[10, 9, 3, 0, 1, 2, 11, 12, 13, 14, 15, 4, 5, 6, 7, 8]
    if name == "vis_pedestrian4":
        order =[10, 0, 1, 2, 3, 11, 12, 13, 4, 5, 6, 7, 8, 9, 14, 15]
    if name == "vis_partylights3":
        order =[10, 9, 8, 7, 1, 0, 2, 3, 11, 12, 13, 14, 15, 4, 5, 6]
    if name == "vis_automobile9":
        order =[4, 11, 5, 10, 6, 9, 7, 8, 2, 13, 0, 1, 3, 12, 14, 15]
    if name == "vis_automobile2":
        order =[5, 6, 7, 8, 14, 0, 1, 4, 15, 2, 3, 9, 10, 11, 12, 13]
    if name == "vis_cloth1":
        order =[5, 2, 3, 4, 6, 11, 12, 13, 14, 15, 0, 1, 7, 8, 9, 10]
    if name == "vis_car6":
        order =[9, 3, 15, 0, 1, 2, 12, 13, 14, 4, 5, 6, 7, 8, 10, 11]
    if name == "vis_whitecup3":
        order =[10, 9, 6, 8, 7, 0, 1, 2, 3, 11, 12, 13, 14, 15, 4, 5]
    if name == "vis_rider4":
        order =[7, 1, 0, 12, 13, 14, 15, 2, 3, 4, 5, 6, 8, 9, 10, 11]
    if name == "vis_car5":
        order =[6, 0, 3, 12, 13, 14, 15, 1, 2, 4, 5, 7, 8, 9, 10, 11]
    if name == "vis_taxi":
        order =[5, 9, 8, 15, 2, 3, 4, 14, 0, 1, 6, 7, 10, 11, 12, 13]
    if name == "vis_rider3":
        order =[10, 9, 8, 2, 11, 0, 1, 3, 14, 15, 4, 5, 6, 7, 12, 13]
    if name == "vis_car4":
        order =[10, 0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 5, 6, 7, 8, 9]
    if name == "vis_officechair1":
        order =[10, 9, 6, 8, 7, 0, 1, 2, 3, 11, 12, 13, 14, 15, 4, 5]
    if name == "vis_bus":
        order =[6, 0, 12, 1, 2, 5, 9, 13, 14, 15, 3, 4, 7, 8, 10, 11]
    if name == "vis_automobile10":
        order =[10, 12, 0, 1, 2, 3, 4, 11, 13, 5, 6, 7, 8, 9, 14, 15]
    if name == "vis_automobile3":
        order =[9, 3, 15, 0, 1, 2, 7, 8, 14, 4, 5, 6, 10, 11, 12, 13]
    if name == "vis_kangaroo":
        order =[11, 5, 6, 7, 8, 14, 2, 3, 4, 12, 13, 15, 0, 1, 9, 10]
    if name == "vis_toy":
        order =[5, 10, 6, 9, 7, 8, 0, 15, 1, 2, 3, 4, 11, 12, 13, 14]
    if name == "vis_board":
        order =[5, 6, 7, 13, 0, 1, 2, 3, 4, 12, 14, 15, 8, 9, 10, 11]
    if name == "vis_rider1":
        order =[10, 0, 1, 2, 3, 11, 12, 13, 4, 5, 6, 7, 8, 9, 14, 15]
    if name == "vis_glass2":
        order =[4, 11, 5, 10, 6, 9, 3, 12, 0, 1, 2, 13, 14, 15, 7, 8]
    if name == "vis_duck3":
        order =[10, 9, 6, 8, 7, 0, 1, 2, 3, 11, 12, 13, 14, 15, 4, 5]
    if name == "vis_rider2":
        order =[4, 11, 5, 10, 6, 9, 7, 8, 0, 1, 2, 3, 12, 13, 14, 15]
    if name == "vis_pedestrian":
        order =[4, 11, 5, 10, 6, 9, 7, 8, 0, 1, 2, 3, 12, 13, 14, 15]
    if name == "vis_automobile6":
        order =[5, 6, 7, 8, 14, 0, 1, 2, 3, 4, 12, 13, 15, 9, 10, 11]
    if name == "vis_car9":
        order =[10, 5, 0, 1, 2, 3, 4, 6, 13, 14, 15, 7, 8, 9, 11, 12]
    if name == "vis_cards11":
        order =[10, 9, 6, 8, 7, 0, 1, 2, 3, 11, 12, 13, 14, 15, 4, 5]
    if name == "vis_ball&mirror7":
        order =[10, 9, 8, 7, 1, 0, 2, 3, 11, 12, 13, 14, 15, 4, 5, 6]
    if name == "vis_bytheriver1":
        order =[10, 9, 8, 2, 0, 1, 3, 11, 12, 13, 14, 15, 4, 5, 6, 7]


    if name == "nir_car18":
        order =[10, 9, 6, 8, 7, 0, 1, 2, 3, 11, 12, 13, 14, 15, 4, 5]
    if name == "nir_car13":
        order =[11, 5, 10, 6, 7, 8, 14, 0, 1, 2, 3, 4, 9, 12, 13, 15]
    if name == "nir_bus3":
        order =[9, 3, 15, 0, 1, 2, 12, 13, 14, 4, 5, 6, 7, 8, 10, 11]
    if name == "nir_car41":
        order =[11, 5, 10, 6, 7, 13, 0, 1, 2, 3, 4, 12, 14, 15, 8, 9]
    if name == "nir_car31":
        order =[5, 10, 6, 7, 0, 1, 4, 11, 12, 13, 2, 3, 8, 9, 14, 15]
    if name == "rednir_duck3":
        order =[10, 9, 6, 8, 7, 0, 1, 2, 3, 11, 12, 13, 14, 15, 4, 5]
    if name == "nir_bus4":
        order =[4, 5, 10, 6, 9, 7, 0, 1, 8, 11, 12, 13, 14, 15, 2, 3]
    if name == "nir_basketball1":
        order =[9, 2, 15, 3, 8, 12, 0, 1, 4, 5, 6, 7, 10, 11, 13, 14]
    if name == "rednir_pool5":
        order =[10, 9, 8, 7, 1, 0, 2, 3, 11, 12, 13, 14, 15, 4, 5, 6]
    if name == "rednir_cards11":
        order =[7, 1, 0, 12, 13, 14, 15, 2, 3, 4, 5, 6, 8, 9, 10, 11]
    if name == "rednir_whitecup3":
        order =[10, 9, 8, 7, 1, 0, 2, 3, 11, 12, 13, 14, 15, 4, 5, 6]
    if name == "nir_rider12":
        order =[10, 5, 11, 0, 1, 2, 3, 4, 12, 13, 6, 7, 8, 9, 14, 15]
    if name == "nir_car36":
        order =[10, 13, 0, 1, 2, 3, 4, 11, 12, 5, 6, 7, 8, 9, 14, 15]
    if name == "nir_car35":
        order =[5, 3, 0, 4, 6, 11, 12, 13, 14, 15, 1, 2, 7, 8, 9, 10]
    if name == "nir_car34":
        order =[4, 11, 5, 10, 6, 9, 7, 8, 0, 1, 2, 3, 12, 13, 14, 15]
    if name == "nir_rider13":
        order =[9, 4, 3, 0, 1, 2, 10, 13, 14, 15, 5, 6, 7, 8, 11, 12]
    if name == "rednir_glass2":
        order =[10, 9, 8, 7, 1, 0, 2, 3, 11, 12, 13, 14, 15, 4, 5, 6]
    if name == "nir_rider7":
        order =[10, 3, 11, 0, 1, 2, 9, 12, 13, 4, 5, 6, 7, 8, 14, 15]
    if name == "nir_car25":
        order =[4, 11, 5, 10, 6, 9, 7, 8, 0, 1, 2, 3, 12, 13, 14, 15]
    if name == "nir_car14":
        order =[7, 8, 0, 1, 2, 13, 14, 15, 3, 4, 5, 6, 9, 10, 11, 12]
    if name == "nir_car27":
        order =[5, 10, 6, 7, 0, 1, 4, 11, 12, 13, 2, 3, 8, 9, 14, 15]
    if name == "nir_car29":
        order =[4, 11, 5, 10, 6, 9, 7, 8, 0, 1, 2, 3, 12, 13, 14, 15]
    if name == "rednir_rainystreet5":
        order =[10, 9, 8, 7, 0, 1, 2, 3, 13, 14, 15, 4, 5, 6, 11, 12]
    if name == "rednir_dice1":
        order =[10, 9, 8, 7, 1, 0, 2, 3, 11, 12, 13, 14, 15, 4, 5, 6]
    if name == "nir_car37":
        order =[4, 11, 5, 10, 6, 9, 7, 8, 0, 1, 2, 3, 12, 13, 14, 15]
    if name == "rednir_bytheriver1":
        order =[4, 11, 5, 10, 6, 9, 7, 8, 0, 1, 2, 3, 12, 13, 14, 15]
    if name == "rednir_ball&mirror7":
        order =[10, 9, 8, 7, 1, 0, 2, 3, 11, 12, 13, 14, 15, 4, 5, 6]
    if name == "nir_car39":
        order =[8, 2, 15, 0, 1, 9, 14, 3, 4, 5, 6, 7, 10, 11, 12, 13]
    if name == "rednir_rainystreet2":
        order =[10, 9, 6, 8, 7, 0, 1, 2, 3, 11, 12, 13, 14, 15, 4, 5]
    if name == "nir_car20":
        order =[4, 11, 5, 10, 6, 9, 7, 8, 0, 1, 2, 3, 12, 13, 14, 15]
    if name == "nir_car21":
        order =[9, 3, 15, 0, 1, 2, 12, 13, 14, 4, 5, 6, 7, 8, 10, 11]
    if name == "rednir_receipts3":
        order =[10, 7, 13, 0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 14, 15]
    if name == "rednir_officechair1":
        order =[10, 8, 0, 9, 13, 14, 15, 1, 2, 3, 4, 5, 6, 7, 11, 12]
    if name == "nir_car16":
        order =[7, 8, 14, 0, 1, 2, 3, 13, 15, 4, 5, 6, 9, 10, 11, 12]
    if name == "nir_rider5":
        order =[4, 11, 5, 10, 6, 9, 7, 8, 0, 1, 2, 3, 12, 13, 14, 15]
    if name == "nir_rider8":
        order =[10, 5, 11, 0, 1, 2, 3, 4, 12, 13, 6, 7, 8, 9, 14, 15]
    if name == "nir_car22":
        order =[10, 0, 1, 2, 3, 11, 12, 13, 4, 5, 6, 7, 8, 9, 14, 15]
    if name == "nir_car23":
        order =[9, 8, 15, 0, 1, 2, 3, 14, 4, 5, 6, 7, 10, 11, 12, 13]
    if name == "rednir_cloth1":
        order =[10, 9, 8, 7, 1, 0, 2, 3, 11, 12, 13, 14, 15, 4, 5, 6]
    if name == "nir_car19":
        order =[9, 8, 2, 4, 0, 1, 3, 5, 12, 13, 14, 15, 6, 7, 10, 11]
    if name == "nir_car26":
        order =[4, 5, 9, 7, 8, 2, 3, 6, 12, 13, 14, 15, 0, 1, 10, 11]
    if name == "nir_car17":
        order =[10, 0, 1, 2, 3, 11, 12, 13, 4, 5, 6, 7, 8, 9, 14, 15]
    if name == "nir_pedestrian5":
        order =[9, 4, 3, 0, 1, 2, 10, 11, 12, 15, 5, 6, 7, 8, 13, 14]
    if name == "nir_car24":
        order =[6, 7, 0, 13, 3, 14, 15, 1, 2, 4, 5, 8, 9, 10, 11, 12]
    if name == "nir_car33":
        order =[10, 9, 8, 2, 0, 1, 3, 11, 12, 13, 14, 15, 4, 5, 6, 7]
    if name == "nir_car30":
        order =[4, 5, 10, 6, 9, 7, 8, 0, 1, 2, 3, 11, 12, 13, 14, 15]
    if name == "nir_car28":
        order =[5, 6, 0, 3, 4, 12, 13, 14, 15, 1, 2, 7, 8, 9, 10, 11]
    if name == "nir_car15":
        order =[6, 7, 11, 0, 13, 1, 12, 14, 15, 2, 3, 4, 5, 8, 9, 10]
    if name == "rednir_partylights3":
        order =[10, 9, 8, 7, 1, 0, 2, 3, 11, 12, 13, 14, 15, 4, 5, 6]
    if name == "nir_rider6":
        order =[5, 10, 0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 6, 7, 8, 9]
    if name == "nir_car32":
        order =[10, 2, 13, 0, 1, 3, 11, 12, 4, 5, 6, 7, 8, 9, 14, 15]
    if name == "rednir_officefan2":
        order =[10, 9, 8, 7, 1, 0, 2, 3, 11, 12, 13, 14, 15, 4, 5, 6]
    if name == "nir_truck1":
        order =[5, 6, 9, 7, 8, 15, 0, 3, 4, 13, 14, 1, 2, 10, 11, 12]
    if name == "nir_car40":
        order =[10, 0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 5, 6, 7, 8, 9]
    if name == "nir_car38":
        order =[4, 11, 5, 10, 6, 9, 7, 8, 0, 1, 2, 3, 12, 13, 14, 15]


    rgb_image1,rgb_image2= HSI2RGB(sample, order)
    rgb_image1 = rgb_image1.transpose(1, 2, 0)
    rgb_image2 = rgb_image2.transpose(1, 2, 0)
    img = np.concatenate((rgb_image1, rgb_image2), axis=2)
    return img
