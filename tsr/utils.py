#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 9 23:11:51 2020
utils
@author: chineseocr
"""
from math import atan, pi
import cv2
import numpy as np
from scipy.ndimage import filters, interpolation
from skimage import measure
from numpy import amin, amax
from scipy.spatial import distance as dist
from numba import jit
from common.ocr_utils import order_points
from common.params import args
from loguru import logger as log
from math import fabs, sin, cos, radians

def warp_img(src_img, kps, cal_reM=True, borderValue = (255, 255, 255)):
    img_crop_width = int(max(np.linalg.norm(kps[0] - kps[1]), np.linalg.norm(kps[2] - kps[3])))
    img_crop_height = int(max(np.linalg.norm(kps[0] - kps[3]), np.linalg.norm(kps[1] - kps[2])))
    short_size = min(img_crop_width, img_crop_height)
    ratio = img_crop_width / img_crop_height
    # ratio = ratio if img_crop_width >= img_crop_height else 1 / ratio
    if ratio > 1:
        obj_h = short_size
        obj_w = int(obj_h * ratio)
    else:
        obj_w = short_size
        obj_h = int(obj_w / ratio)

    input_pts = np.float32([kps[0], kps[1], kps[2], kps[3]])
    # output_pts = np.float32([[0, 0], [obj_w - 1, 0],
    #                          [obj_w - 1, obj_h - 1], [0, obj_h - 1]])
    output_pts = np.float32([[kps[0][0], kps[0][1]], [obj_w+kps[0][0], kps[0][1]],
                             [obj_w+kps[0][0], obj_h+kps[0][1]], [kps[0][0], obj_h+kps[0][1]]])
    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    if cal_reM:
        reM = cv2.getPerspectiveTransform(output_pts, input_pts)  # TODO
    else:
        reM = None
    # obj_img = cv2.warpPerspective(src_img, M, (obj_w, obj_h))
    obj_img = cv2.warpPerspective(src_img, M, (src_img.shape[1], src_img.shape[0]),
                                  borderValue = borderValue,
                                  flags = cv2.INTER_CUBIC
                                  )
    return obj_img, M, reM

def get_img_rot_broa(img, degree=90, cal_reM=True, borderValue=(255, 255, 255), min_angle=0.0):
    # if abs(degree) < min_angle or abs(degree) > 90 - min_angle:
    #     return img, None, None

    height, width = img.shape[:2]
    height_new = int(width * fabs(sin(radians(degree))) +
                     height * fabs(cos(radians(degree))))
    width_new = int(height * fabs(sin(radians(degree))) +
                    width * fabs(cos(radians(degree))))
    mat_rotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    # inv_mat_rotation = np.linalg.pinv(mat_rotation)
    mat_rotation[0, 2] += (width_new - width) / 2
    mat_rotation[1, 2] += (height_new - height) / 2
    img_rotated = cv2.warpAffine(img, mat_rotation, (width_new, height_new),
                                 borderValue=borderValue)
    if cal_reM:
        inv_mat_rotation = np.copy(mat_rotation) * -1
        inv_mat_rotation[0,0] *= -1
        inv_mat_rotation[1,1] *= -1
    else:
        inv_mat_rotation = None
    return img_rotated, mat_rotation, inv_mat_rotation

def reCalculateAffine(BBS, M):
    # TODO 有问题，丢弃
    BBS = np.array(BBS, dtype = 'float32')
    if M is None:
        return BBS
    if BBS.ndim == 2:
        for i in range(len(BBS)):
            x0, y0, x1, y1, x2, y2, x3, y3 = BBS[i]
            x0 = M[0][0] * x0 + M[0][1] * y0 + M[0][2]
            y0 = M[1][0] * x0 + M[1][1] * y0 + M[1][2]
            x1 = M[0][0] * x1 + M[0][1] * y1 + M[0][2]
            y1 = M[1][0] * x1 + M[1][1] * y1 + M[1][2]
            x2 = M[0][0] * x2 + M[0][1] * y2 + M[0][2]
            y2 = M[1][0] * x2 + M[1][1] * y2 + M[1][2]
            x3 = M[0][0] * x3 + M[0][1] * y3 + M[0][2]
            y3 = M[1][0] * x3 + M[1][1] * y3 + M[1][2]
            BBS[i] = [int(x0), int(y0), int(x1), int(y1), int(x2), int(y2), int(x3), int(y3)]
    elif BBS.ndim == 1:
        x0, y0, x1, y1, x2, y2, x3, y3 = BBS
        x0 = M[0][0] * x0 + M[0][1] * y0 + M[0][2]
        y0 = M[1][0] * x0 + M[1][1] * y0 + M[1][2]
        x1 = M[0][0] * x1 + M[0][1] * y1 + M[0][2]
        y1 = M[1][0] * x1 + M[1][1] * y1 + M[1][2]
        x2 = M[0][0] * x2 + M[0][1] * y2 + M[0][2]
        y2 = M[1][0] * x2 + M[1][1] * y2 + M[1][2]
        x3 = M[0][0] * x3 + M[0][1] * y3 + M[0][2]
        y3 = M[1][0] * x3 + M[1][1] * y3 + M[1][2]
        BBS = np.array([int(x0), int(y0), int(x1), int(y1), int(x2), int(y2), int(x3), int(y3)], dtype = 'float32')
    return BBS


def get_rotate_crop_image(img, points, cal_reM=True, borderValue = (255, 255, 255)):
    # points = order_points(points)
    img_crop_width = int(max(np.linalg.norm(points[0] - points[1]), np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(max(np.linalg.norm(points[0] - points[3]), np.linalg.norm(points[1] - points[2])))

    scale = img_crop_width / img_crop_height
    if scale < 1:
        top_left_x = img_crop_width // 10
        top_left_y = img_crop_height // 10
    else:
        top_left_x = img_crop_height // 10
        top_left_y = img_crop_width // 10
    new_box_h = int((img_crop_width - top_left_x - top_left_x) / scale)
    pts_std = np.float32([[top_left_x, top_left_y], [img_crop_width - top_left_x, top_left_y],
                          [img_crop_width - top_left_x, new_box_h + top_left_y],
                          [top_left_x, new_box_h + top_left_y]])
    # pts_std = np.array([
    #     [0, 0],
    #     [img_crop_width - 1, 0],
    #     [img_crop_width - 1, img_crop_height - 1],
    #     [0, img_crop_height - 1]],
    #     dtype = 'float32')
    M = cv2.getPerspectiveTransform(points, pts_std)
    if cal_reM:
        reM = cv2.getPerspectiveTransform(pts_std, points)  # TODO
    else:
        reM = None
    new_img_crop_height = new_box_h + top_left_y + top_left_y
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, new_img_crop_height),
        borderValue = borderValue,
        flags=cv2.INTER_CUBIC)
    # dst_img = cv2.warpPerspective(img, M, (img_crop_width, img_crop_height))
    return dst_img, M, reM

def reCalculatePerspective(BBS, M):
    '''
    计算出透视变换的变换矩阵之后，准备对一组关键点进行变换，以得到关键点变换之后的位置。
    :param BBS: 原始坐标点
    :param M: 变换矩阵
    :return: 新的坐标点
    '''
    if M is None:
        return BBS
    BBS = np.array(BBS, dtype = 'float32')
    if BBS.ndim == 2:
        for i in range(len(BBS)):
            # x0, y0, x1, y1, x2, y2, x3, y3 = BBS[i]
            # x = (M[0][0]*x0+M[0][1]*y0+M[0][2]) / (M[2][0]*x0+M[2][1]*y0+M[2][2])
            # y = (M[1][0]*x0+M[1][1]*y0+M[1][2]) / (M[2][0]*x0+M[2][1]*y0+M[2][2])
            # x = (M[0][0]*x1+M[0][1]*y1+M[0][2]) / (M[2][0]*x1+M[2][1]*y1+M[2][2])
            # y = (M[1][0]*x1+M[1][1]*y1+M[1][2]) / (M[2][0]*x1+M[2][1]*y1+M[2][2])
            points = np.array(BBS[i])
            points = np.array(points).reshape(1, -1, 2).astype(np.float32)
            new_points = cv2.perspectiveTransform(points, M)
            BBS[i] = new_points.astype('int').flatten()
    elif BBS.ndim == 1:
        BBS = np.array(BBS).reshape(-1, 1, 2).astype(np.float32)
        BBS = cv2.perspectiveTransform(BBS, M).flatten()
    return BBS

# def edge_detection(seg_map):
#     # 1. edge detection
#     seg_map = seg_map.copy()
#     # seg_map = cv2.GaussianBlur(seg_map, (5, 5), 0)
#     # seg_map = cv2.Canny(seg_map, 75, 200)
#     cnts = cv2.findContours(seg_map.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     if len(cnts) == 2:
#         cnts = cnts[0]
#     elif len(cnts) == 3:
#         cnts = cnts[1]
#     cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
#
#     screenCnt = None
#     for c in cnts:
#         contourArea = cv2.contourArea(c)
#         if contourArea < seg_map.shape[0] * seg_map.shape[1] * 0.05:
#             continue
#         peri = cv2.arcLength(c, True)
#         # approx = cv2.approxPolyDP(c, 0.025 * peri, True) # 设定的阈值越小，拟合的越精准，拟合后多边形的边和顶点越多
#         for threshold in np.arange(0.025, 0.06, 0.01):
#             threshold = round(threshold, 3)
#             # print(threshold)
#             approx = cv2.approxPolyDP(c, threshold * peri, True)
#             if len(approx) == 4:
#                 screenCnt = approx
#                 break
#         break
#     if args.table_save and screenCnt is not None:
#         tmp = seg_map.copy() * 255
#         tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
#         cv2.drawContours(tmp, [screenCnt], -1, (0, 0, 255), 2)
#         cv2.imwrite('test/contours.jpg', tmp)
#     if screenCnt is not None:
#         screenCnt = order_points(screenCnt.reshape(4, 2))
#         row1 = (screenCnt[1][1] - screenCnt[0][1]) / (screenCnt[1][0] - screenCnt[0][0])
#         row2 = (screenCnt[2][1] - screenCnt[3][1]) / (screenCnt[2][0] - screenCnt[3][0])
#         col1 = (screenCnt[3][1] - screenCnt[0][1]) / (screenCnt[3][0] - screenCnt[0][0])
#         col2 = (screenCnt[2][1] - screenCnt[1][1]) / (screenCnt[2][0] - screenCnt[1][0])
#         degree_row1 = atan(row1) * 180 / pi
#         degree_row2 = atan(row2) * 180 / pi
#         degree_col1 = atan(col1) * 180 / pi
#         degree_col2 = atan(col2) * 180 / pi
#         diff_degree_row1 = 0 - degree_row1
#         diff_degree_row2 = 0 - degree_row2
#         if degree_col1 > 0:
#             diff_degree_col1 = 90 - degree_col1
#         else:
#             diff_degree_col1 = -90 - degree_col1
#         if degree_col2 > 0:
#             diff_degree_col2 = 90 - degree_col2
#         else:
#             diff_degree_col2 = -90 - degree_col2
#         min_diff_degree_row = min(abs(diff_degree_row1), abs(diff_degree_row2))
#         max_diff_degree_row = max(abs(diff_degree_row1), abs(diff_degree_row2))
#         min_diff_degree_col = min(abs(diff_degree_col1), abs(diff_degree_col2))
#         max_diff_degree_col = max(abs(diff_degree_col1), abs(diff_degree_col2))
#         if abs(diff_degree_row1 - diff_degree_row2) > 10 or abs(diff_degree_col1 - diff_degree_col2) > 50:
#                 screenCnt = None
#         if (abs(min_diff_degree_row) < 1 and abs(max_diff_degree_row) > 5) or \
#                 (abs(min_diff_degree_col) < 1 and abs(max_diff_degree_col) > 25):
#             screenCnt = None
#     return screenCnt

def get_angle(sta_point, mid_point, end_point):
    ma_x = sta_point[0]-mid_point[0]
    ma_y = sta_point[1]-mid_point[1]
    mb_x = end_point[0]-mid_point[0]
    mb_y = end_point[1]-mid_point[1]
    ab_x = sta_point[0]-end_point[0]
    ab_y = sta_point[1]-end_point[1]
    ab_val2 = ab_x * ab_x + ab_y * ab_y
    ma_val2 = ma_x * ma_x + ma_y * ma_y
    mb_val2 = mb_x * mb_x + mb_y * mb_y
    cos_M = (ma_val2+mb_val2-ab_val2) / (2 * np.sqrt(ma_val2)*np.sqrt(mb_val2))
    angleAMB = np.arccos(cos_M)/np.pi * 180
    return angleAMB

def get_angles(screenCnt, polygon=4):
    if isinstance(screenCnt, list):
        screenCnt_ = screenCnt*2
    else:
        screenCnt_ = screenCnt.tolist()*2
    edge_list = [screenCnt_[i:i+polygon-1] for i in range(4)]
    angle_list = [get_angle(*i) for i in edge_list]
    return angle_list

def edge_detection_(seg_map):
    def find_vertex(c_temp, x1, y1, x2, y2, max_c_list=[]):
        max_dis = 0
        for i in c_temp.tolist():
            if i in max_c_list:
                continue
            dis = np.sqrt((x1 - i[0]) ** 2 + (y1 - i[1]) ** 2)
            if x2 and y2:
                dis2 = np.sqrt((x2 - i[0]) ** 2 + (y2 - i[1]) ** 2)
                dis += dis2
            if dis > max_dis:
                max_dis = dis
                max_c = i
        if max_c not in max_c_list:
            max_c_list.append(max_c)

    # 1. edge detection
    seg_map = seg_map.copy()
    # seg_map = cv2.GaussianBlur(seg_map, (5, 5), 0)
    # seg_map = cv2.Canny(seg_map, 75, 200)
    cnts = cv2.findContours(seg_map.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 2:
        cnts = cnts[0]
    elif len(cnts) == 3:
        cnts = cnts[1]
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    screenCnt = None
    for c in cnts:
        contourArea = cv2.contourArea(c)
        if contourArea < seg_map.shape[0] * seg_map.shape[1] * 0.05:
            continue
        M = cv2.moments(c)
        center_x = int(M['m10'] / M['m00'])
        center_y = int(M['m01'] / M['m00'])
        c_temp = c.squeeze()
        screenCnt = []
        find_vertex(c_temp, center_x, center_y, None, None, max_c_list = screenCnt)
        find_vertex(c_temp, screenCnt[-1][0], screenCnt[-1][1], None, None, max_c_list = screenCnt)
        find_vertex(c_temp, screenCnt[-1][0], screenCnt[-1][1], screenCnt[-2][0], screenCnt[-2][1], max_c_list = screenCnt)
        find_vertex(c_temp, screenCnt[-1][0], screenCnt[-1][1], None, None, max_c_list = screenCnt)
        x_min, x_max = np.array(screenCnt)[:,0].min(), np.array(screenCnt)[:,0].max()
        y_min, y_max = np.array(screenCnt)[:,1].min(), np.array(screenCnt)[:,1].max()
        diff = 20
        if x_min-0<diff or seg_map.shape[1]-x_max<diff or y_min-0<diff or seg_map.shape[0]-y_max<diff:
            flag_diff = False
        else:
            flag_diff = True
        if len(screenCnt) == 4 and flag_diff:
            _, (w, h), _ = cv2.minAreaRect(np.int32(screenCnt))
            if w>20 and h>20:
                screenCnt = order_points(np.array(screenCnt).reshape(4, 2))
                break
        else:
            screenCnt = None

    if args.table_save and screenCnt is not None:
        tmp = seg_map.copy() * 255
        tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(tmp, [np.int32(screenCnt).reshape(4,-1,2)], -1, (0, 0, 255), 2)
        cv2.imwrite('test/contours.jpg', tmp)
    if screenCnt is not None:
        row1 = (screenCnt[1][1] - screenCnt[0][1]) / (screenCnt[1][0] - screenCnt[0][0])
        row2 = (screenCnt[2][1] - screenCnt[3][1]) / (screenCnt[2][0] - screenCnt[3][0])
        col1 = (screenCnt[3][1] - screenCnt[0][1]) / (screenCnt[3][0] - screenCnt[0][0])
        col2 = (screenCnt[2][1] - screenCnt[1][1]) / (screenCnt[2][0] - screenCnt[1][0])
        degree_row1 = atan(row1) * 180 / pi
        degree_row2 = atan(row2) * 180 / pi
        degree_col1 = atan(col1) * 180 / pi
        degree_col2 = atan(col2) * 180 / pi
        diff_degree_row1 = 0 - degree_row1
        diff_degree_row2 = 0 - degree_row2
        if degree_col1 > 0:
            diff_degree_col1 = 90 - degree_col1
        else:
            diff_degree_col1 = -90 - degree_col1
        if degree_col2 > 0:
            diff_degree_col2 = 90 - degree_col2
        else:
            diff_degree_col2 = -90 - degree_col2
        min_diff_degree_row = min(abs(diff_degree_row1), abs(diff_degree_row2))
        max_diff_degree_row = max(abs(diff_degree_row1), abs(diff_degree_row2))
        min_diff_degree_col = min(abs(diff_degree_col1), abs(diff_degree_col2))
        max_diff_degree_col = max(abs(diff_degree_col1), abs(diff_degree_col2))
        # if abs(diff_degree_row1 - diff_degree_row2) > 10 or abs(diff_degree_col1 - diff_degree_col2) > 50:
        #         screenCnt = None
        # if (abs(min_diff_degree_row) < 1 and abs(max_diff_degree_row) > 5) or \
        #         (abs(min_diff_degree_col) < 1 and abs(max_diff_degree_col) > 25):
        #     screenCnt = None
        if abs(diff_degree_row1 - diff_degree_row2) > 25 or abs(diff_degree_col1 - diff_degree_col2) > 50:
                screenCnt = None
        if (abs(min_diff_degree_row) < 2 and abs(max_diff_degree_row) > 5) or \
                (abs(min_diff_degree_col) < 2 and abs(max_diff_degree_col) > 25):
            screenCnt = None
        if screenCnt is not None:
            angle_list = get_angles(screenCnt, polygon = 4)
            max_angle, min_angle = max(angle_list), min(angle_list)
            if max_angle - min_angle > 30:
                screenCnt = None
            # dis_90 = [abs(int(90-i)) for i in angle_list]
            # if 1<=dis_90.count(0)<=3:
            #     screenCnt = None
    return screenCnt

def initial_verify_corner(screenCnt):
    if screenCnt is not None:
        screenCnt = order_points(screenCnt.reshape(4, 2))
        row1 = (screenCnt[1][1] - screenCnt[0][1]) / (screenCnt[1][0] - screenCnt[0][0])
        row2 = (screenCnt[2][1] - screenCnt[3][1]) / (screenCnt[2][0] - screenCnt[3][0])
        col1 = (screenCnt[3][1] - screenCnt[0][1]) / (screenCnt[3][0] - screenCnt[0][0])
        col2 = (screenCnt[2][1] - screenCnt[1][1]) / (screenCnt[2][0] - screenCnt[1][0])
        degree_row1 = atan(row1) * 180 / pi
        degree_row2 = atan(row2) * 180 / pi
        degree_col1 = atan(col1) * 180 / pi
        degree_col2 = atan(col2) * 180 / pi
        diff_degree_row1 = 0 - degree_row1
        diff_degree_row2 = 0 - degree_row2
        if degree_col1 > 0:
            diff_degree_col1 = 90 - degree_col1
        else:
            diff_degree_col1 = -90 - degree_col1
        if degree_col2 > 0:
            diff_degree_col2 = 90 - degree_col2
        else:
            diff_degree_col2 = -90 - degree_col2
        min_diff_degree_row = min(abs(diff_degree_row1), abs(diff_degree_row2))
        max_diff_degree_row = max(abs(diff_degree_row1), abs(diff_degree_row2))
        min_diff_degree_col = min(abs(diff_degree_col1), abs(diff_degree_col2))
        max_diff_degree_col = max(abs(diff_degree_col1), abs(diff_degree_col2))
        if abs(diff_degree_row1 - diff_degree_row2) > 25 or abs(diff_degree_col1 - diff_degree_col2) > 50:
                screenCnt = None
        if (abs(min_diff_degree_row) < 2 and abs(max_diff_degree_row) > 5) or \
                (abs(min_diff_degree_col) < 2 and abs(max_diff_degree_col) > 25):
            screenCnt = None
        if screenCnt is not None:
            angle_list = get_angles(screenCnt, polygon = 4)
            max_angle, min_angle = max(angle_list), min(angle_list)
            if max_angle - min_angle > 30:
                screenCnt = None
            # dis_90 = [abs(int(90-i)) for i in angle_list]
            # if 1<=dis_90.count(0)<=3:
            #     screenCnt = None
    return screenCnt

def edge_detection(seg_map):
    # 1. edge detection
    seg_map = seg_map.copy()
    # seg_map = cv2.GaussianBlur(seg_map, (5, 5), 0)
    # seg_map = cv2.Canny(seg_map, 75, 200)
    cnts = cv2.findContours(seg_map.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 2:
        cnts = cnts[0]
    elif len(cnts) == 3:
        cnts = cnts[1]
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]

    screenCnt = None
    for c in cnts:
        contourArea = cv2.contourArea(c)
        if contourArea < seg_map.shape[0] * seg_map.shape[1] * 0.05:
            continue
        peri = cv2.arcLength(c, True)
        # approx = cv2.approxPolyDP(c, 0.025 * peri, True) # 设定的阈值越小，拟合的越精准，拟合后多边形的边和顶点越多
        for threshold in np.arange(0.025, 0.45, 0.01):
            threshold = round(threshold, 3)
            # print(threshold)
            approx = cv2.approxPolyDP(c, threshold * peri, True)
            if len(approx) == 4:
                screenCnt = approx
                break
        break
    if args.table_save and screenCnt is not None:
        tmp = seg_map.copy() * 255
        tmp = cv2.cvtColor(tmp, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(tmp, [screenCnt], -1, (0, 0, 255), 2)
        cv2.imwrite('test/contours.jpg', tmp)

    screenCnt = initial_verify_corner(screenCnt)
    return screenCnt

def transform(ori_img, screenCnt, final_rboxes=[], cal_reM=True, borderValue = (255, 255, 255), print_log=True):
    if screenCnt is not None:
        # cv2.drawContours(seg_map, [screenCnt], -1, (0, 255, 0), 2)
        warped, M, reM = get_rotate_crop_image(ori_img, screenCnt, cal_reM, borderValue)
        if print_log:
            log.info("透视变换成功")
        return 'perspective', warped, M, reM
    elif len(final_rboxes) >= 3:
        RowsLines = []
        for box in final_rboxes:
            RowsLines.append([box[0], box[1], box[2], box[3]])
            RowsLines.append([box[6], box[7], box[4], box[5]])
        rotate_angle = np.array([np.arctan2(rowline[3] - rowline[1], rowline[2] - rowline[0]) for rowline in RowsLines])
        rotate_angle = sorted(rotate_angle)
        rotate_angle = np.array(rotate_angle[1:-1])
        rotate_angle = np.mean(rotate_angle * 180.0 / np.pi)
        if abs(rotate_angle) > 0.01:
            warped, M, reM = get_img_rot_broa(ori_img, rotate_angle, cal_reM, borderValue, 0.0)
            if print_log:
                log.info("仿射变换成功")
            return 'affine', warped, rotate_angle, None #M, reM
        else:
            if print_log:
                log.info("无需进行仿射变换")
            return None, ori_img, None, None
    else:
        if print_log:
            log.info("没有可见清晰边缘，透视变换失败")
        return None, ori_img, None, None

def nms_box(boxes, scores, score_threshold=0.5, nms_threshold=0.3):
    ##nms box
    boxes = np.array(boxes)
    scores = np.array(scores)
    ind = scores > score_threshold
    boxes = boxes[ind]
    scores = scores[ind]

    def box_to_center(box):
        xmin, ymin, xmax, ymax = [round(float(x), 4) for x in box]
        w = xmax - xmin
        h = ymax - ymin
        return [round(xmin, 4), round(ymin, 4), round(w, 4), round(h, 4)]

    newBoxes = [box_to_center(box) for box in boxes]
    newscores = [round(float(x), 6) for x in scores]

    index = cv2.dnn.NMSBoxes(newBoxes, newscores, score_threshold=score_threshold, nms_threshold=nms_threshold)
    if len(index) > 0:
        index = index.reshape((-1,))
        return boxes[index], scores[index]
    else:
        return np.array([]), np.array([])


def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, (0, 0), fx=f, fy=f)


def estimate_skew_angle(raw, angleRange=[-15, 15]):
    """
    估计图像文字偏转角度,
    angleRange:角度估计区间
    """
    raw = resize_im(raw, scale=600, max_scale=900)
    image = raw - amin(raw)
    image = image / amax(image)
    m = interpolation.zoom(image, 0.5)
    m = filters.percentile_filter(m, 80, size=(20, 2))
    m = filters.percentile_filter(m, 80, size=(2, 20))
    m = interpolation.zoom(m, 1.0 / 0.5)
    # w,h = image.shape[1],image.shape[0]
    w, h = min(image.shape[1], m.shape[1]), min(image.shape[0], m.shape[0])
    flat = np.clip(image[:h, :w] - m[:h, :w] + 1, 0, 1)
    d0, d1 = flat.shape
    o0, o1 = int(0.1 * d0), int(0.1 * d1)
    flat = amax(flat) - flat
    flat -= amin(flat)
    est = flat[o0:d0 - o0, o1:d1 - o1]
    angles = range(angleRange[0], angleRange[1])
    estimates = []
    for a in angles:
        roest = interpolation.rotate(est, a, order=0, mode='constant')
        v = np.mean(roest, axis=1)
        v = np.var(v)
        estimates.append((v, a))

    _, a = max(estimates)
    return a


def eval_angle(img, angleRange=[-5, 5]):
    """
    估计图片文字的偏移角度
    """
    im = Image.fromarray(img)
    degree = estimate_skew_angle(np.array(im.convert('L')), angleRange=angleRange)
    im = im.rotate(degree, center=(im.size[0] / 2, im.size[1] / 2), expand=1, fillcolor=(255, 255, 255))
    img = np.array(im)
    return img, degree


def letterbox_image(image, size, fillValue=[128, 128, 128]):
    '''
    resize image with unchanged aspect ratio using padding
    '''
    image_h, image_w = image.shape[:2]
    w, h = size
    new_w = int(image_w * min(w * 1.0 / image_w, h * 1.0 / image_h))
    new_h = int(image_h * min(w * 1.0 / image_w, h * 1.0 / image_h))

    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    # cv2.imwrite('tmp/test.png', resized_image[...,::-1])
    if fillValue is None:
        fillValue = [int(x.mean()) for x in cv2.split(np.array(image))]
    boxed_image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    boxed_image[:] = fillValue
    boxed_image[:new_h, :new_w, :] = resized_image

    return boxed_image, new_w / image_w, new_h / image_h

# def letterbox_image(img, size, fillValue=(128, 128, 128)):
#     '''resize image with unchanged aspect ratio using padding'''
#     image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     image_w, image_h = image.size
#     w, h = size
#     new_w = int(image_w * min(w*1.0/image_w, h*1.0/image_h))
#     new_h = int(image_h * min(w*1.0/image_w, h*1.0/image_h))
#     resized_image = image.resize((new_w,new_h), Image.BICUBIC)
#     fx = new_w/image_w
#     fy = new_h/image_h
#     dx = (w-new_w)//2
#     dy = (h-new_h)//2
#
#     if fillValue is None:
#         fillValue = [int(x.mean()) for x in cv2.split(np.array(image))]
#     boxed_image = Image.new('RGB', size, fillValue)
#     boxed_image.paste(resized_image, (dx,dy))
#     return np.array(boxed_image),fx,fy,dx,dy


def get_table_line(binimg, axis=0, lineW=10, angle = 20):
    ##获取表格线
    ##axis=0 横线
    ##axis=1 竖线
    H, W = binimg.shape
    labels = measure.label(binimg > 0, connectivity=2)  # 8连通区域标记
    regions_ = measure.regionprops(labels)
    regions = []
    for region in regions_:
        if region.bbox_area > H * W * 3 / 4:  # 过滤大的单元格
            continue
        regions.append(region)
    if axis == 1:
        lineboxes = [minAreaRect(line.coords) for line in regions if line.bbox[2] - line.bbox[0] > lineW]
        # lineboxes = [[x1, y1, x2, y2] for (x1, y1, x2, y2) in lineboxes if
        #              (90-(atan(abs((y2 - y1) / (x2 - x1 + 1e-10))) * 180 / pi)) < angle]
    else:
        lineboxes = [minAreaRect(line.coords) for line in regions if line.bbox[3] - line.bbox[1] > lineW]
        # lineboxes = [[x1, y1, x2, y2] for (x1, y1, x2, y2) in lineboxes if
        #              atan(abs((y2 - y1) / (x2 - x1 + 1e-10))) * 180 / pi < angle]
    # lineboxes = [i for i in lineboxes if len(i)>0]
    return lineboxes


def sqrt(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def adjust_lines(RowsLines, ColsLines, row_alph=50, col_alph=50, angle=50):
    # 调整line
    # 两个横线或者两个竖线的任意端点的距离，如横线1的终点与横线2的起点之间的距离，若该距离小于alpha，则认为这两个点可连成一条线

    nrow = len(RowsLines)
    ncol = len(ColsLines)
    newRowsLines = []
    newColsLines = []
    for i in range(nrow):

        x1, y1, x2, y2 = RowsLines[i]
        cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
        for j in range(nrow):
            if i != j:
                x3, y3, x4, y4 = RowsLines[j]
                cx2, cy2 = (x3 + x4) / 2, (y3 + y4) / 2
                if (x3 < cx1 < x4 or y3 < cy1 < y4) or (x1 < cx2 < x2 or y1 < cy2 < y2): # 判断两个横线在y方向的投影重不重合
                    continue
                else:
                    r = sqrt((x1, y1), (x3, y3))
                    k = abs((y3-y1) / (x3-x1+1e-10))
                    a = atan(k) * 180 / pi
                    if r < row_alph and a < angle:
                        newRowsLines.append((x1, y1, x3, y3))

                    r = sqrt((x1, y1), (x4, y4))
                    k = abs((y4 - y1) / (x4 - x1 + 1e-10))
                    a = atan(k) * 180 / pi
                    if r < row_alph and a < angle:
                        newRowsLines.append((x1, y1, x4, y4))

                    r = sqrt((x2, y2), (x3, y3))
                    k = abs((y3 - y2) / (x3 - x2 + 1e-10))
                    a = atan(k) * 180 / pi
                    if r < row_alph and a < angle:
                        newRowsLines.append((x2, y2, x3, y3))
                    r = sqrt((x2, y2), (x4, y4))
                    k = abs((y4 - y2) / (x4 - x2 + 1e-10))
                    a = atan(k) * 180 / pi
                    if r < row_alph and a < angle:
                        newRowsLines.append((x2, y2, x4, y4))

    for i in range(ncol):
        x1, y1, x2, y2 = ColsLines[i]
        cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
        for j in range(ncol):
            if i != j:
                x3, y3, x4, y4 = ColsLines[j]
                cx2, cy2 = (x3 + x4) / 2, (y3 + y4) / 2
                if (x3 < cx1 < x4 or y3 < cy1 < y4) or (x1 < cx2 < x2 or y1 < cy2 < y2):
                    continue
                else:
                    r = sqrt((x1, y1), (x3, y3))
                    k = abs((y3 - y1) / (x3 - x1 + 1e-10))
                    a = atan(k) * 180 / pi
                    if r < col_alph and abs(90-a) < angle:
                        newColsLines.append((x1, y1, x3, y3))
                    r = sqrt((x1, y1), (x4, y4))
                    k = abs((y4 - y1) / (x4 - x1 + 1e-10))
                    a = atan(k) * 180 / pi
                    if r < col_alph and abs(90-a) < angle:
                        newColsLines.append((x1, y1, x4, y4))

                    r = sqrt((x2, y2), (x3, y3))
                    k = abs((y3 - y2) / (x3 - x2 + 1e-10))
                    a = atan(k) * 180 / pi
                    if r < col_alph and abs(90-a) < angle:
                        newColsLines.append((x2, y2, x3, y3))
                    r = sqrt((x2, y2), (x4, y4))
                    k = abs((y4 - y2) / (x4 - x2 + 1e-10))
                    a = atan(k) * 180 / pi
                    if r < col_alph and abs(90-a) < angle:
                        newColsLines.append((x2, y2, x4, y4))

    # return np.array(newRowsLines, dtype = 'float32'), np.array(newColsLines, dtype = 'float32')
    return newRowsLines, newColsLines

@jit(nopython = True)
def numba_sqrt(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

@jit(nopython = True)
def numba_adjust_lines(RowsLines, ColsLines, row_alph=50, col_alph=50, angle=50):
    # 调整line
    # 两个横线或者两个竖线的任意端点的距离，如横线1的终点与横线2的起点之间的距离，若该距离小于alpha，则认为这两个点可连成一条线

    nrow = len(RowsLines)
    ncol = len(ColsLines)
    newRowsLines = []
    newColsLines = []
    for i in range(nrow):

        x1, y1, x2, y2 = RowsLines[i]
        cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
        for j in range(nrow):
            if i != j:
                x3, y3, x4, y4 = RowsLines[j]
                cx2, cy2 = (x3 + x4) / 2, (y3 + y4) / 2
                if (x3 < cx1 < x4 or y3 < cy1 < y4) or (x1 < cx2 < x2 or y1 < cy2 < y2): # 判断两个横线在y方向的投影重不重合
                    continue
                else:
                    r = numba_sqrt((x1, y1), (x3, y3))
                    k = abs((y3-y1) / (x3-x1+1e-10))
                    a = atan(k) * 180 / pi
                    if r < row_alph and a < angle:
                        newRowsLines.append((x1, y1, x3, y3))

                    r = numba_sqrt((x1, y1), (x4, y4))
                    k = abs((y4 - y1) / (x4 - x1 + 1e-10))
                    a = atan(k) * 180 / pi
                    if r < row_alph and a < angle:
                        newRowsLines.append((x1, y1, x4, y4))

                    r = numba_sqrt((x2, y2), (x3, y3))
                    k = abs((y3 - y2) / (x3 - x2 + 1e-10))
                    a = atan(k) * 180 / pi
                    if r < row_alph and a < angle:
                        newRowsLines.append((x2, y2, x3, y3))
                    r = numba_sqrt((x2, y2), (x4, y4))
                    k = abs((y4 - y2) / (x4 - x2 + 1e-10))
                    a = atan(k) * 180 / pi
                    if r < row_alph and a < angle:
                        newRowsLines.append((x2, y2, x4, y4))

    for i in range(ncol):
        x1, y1, x2, y2 = ColsLines[i]
        cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
        for j in range(ncol):
            if i != j:
                x3, y3, x4, y4 = ColsLines[j]
                cx2, cy2 = (x3 + x4) / 2, (y3 + y4) / 2
                if (x3 < cx1 < x4 or y3 < cy1 < y4) or (x1 < cx2 < x2 or y1 < cy2 < y2):
                    continue
                else:
                    r = numba_sqrt((x1, y1), (x3, y3))
                    k = abs((y3 - y1) / (x3 - x1 + 1e-10))
                    a = atan(k) * 180 / pi
                    if r < col_alph and abs(90-a) < angle:
                        newColsLines.append((x1, y1, x3, y3))
                    r = numba_sqrt((x1, y1), (x4, y4))
                    k = abs((y4 - y1) / (x4 - x1 + 1e-10))
                    a = atan(k) * 180 / pi
                    if r < col_alph and abs(90-a) < angle:
                        newColsLines.append((x1, y1, x4, y4))

                    r = numba_sqrt((x2, y2), (x3, y3))
                    k = abs((y3 - y2) / (x3 - x2 + 1e-10))
                    a = atan(k) * 180 / pi
                    if r < col_alph and abs(90-a) < angle:
                        newColsLines.append((x2, y2, x3, y3))
                    r = numba_sqrt((x2, y2), (x4, y4))
                    k = abs((y4 - y2) / (x4 - x2 + 1e-10))
                    a = atan(k) * 180 / pi
                    if r < col_alph and abs(90-a) < angle:
                        newColsLines.append((x2, y2, x4, y4))

    # return np.array(newRowsLines, dtype = 'float32'), np.array(newColsLines, dtype = 'float32')
    return newRowsLines, newColsLines

def filter_lines(RowsLines, ColsLines, angle):
    nrow = len(RowsLines)
    ncol = len(ColsLines)
    newRowsLines = []
    newColsLines = []
    for i in range(nrow):
        x1, y1, x2, y2 = RowsLines[i]
        k = abs((y2 - y1) / (x2 - x1 + 1e-10))
        a = atan(k) * 180 / pi
        if a < angle:
            newRowsLines.append(RowsLines[i])
    for i in range(ncol):
        x1, y1, x2, y2 = ColsLines[i]
        k = abs((y2 - y1) / (x2 - x1 + 1e-10))
        a = atan(k) * 180 / pi
        if abs(90-a) < angle:
            newColsLines.append(ColsLines[i])
    return newRowsLines, newColsLines


def minAreaRect(coords):
    """
    多边形外接矩形
    """
    rect = cv2.minAreaRect(coords[:, ::-1])
    # angle = rect[-1]
    # t = 20 # todo
    # if 90-angle < t or angle < t:
    box = cv2.boxPoints(rect)
    box = box.reshape((8,)).tolist()

    box = image_location_sort_box(box)

    x1, y1, x2, y2, x3, y3, x4, y4 = box
    degree, w, h, cx, cy = solve(box)
    if w < h:
        xmin = (x1 + x2) / 2
        xmax = (x3 + x4) / 2
        ymin = (y1 + y2) / 2
        ymax = (y3 + y4) / 2

    else:
        xmin = (x1 + x4) / 2
        xmax = (x2 + x3) / 2
        ymin = (y1 + y4) / 2
        ymax = (y2 + y3) / 2
    # degree,w,h,cx,cy = solve(box)
    # x1,y1,x2,y2,x3,y3,x4,y4 = box
    # return {'degree':degree,'w':w,'h':h,'cx':cx,'cy':cy}
    return [xmin, ymin, xmax, ymax]


def fit_line(p):
    """A = Y2 - Y1
       B = X1 - X2
       C = X2*Y1 - X1*Y2
       AX+BY+C=0
    直线一般方程
    """
    x1, y1 = p[0]
    x2, y2 = p[1]
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return A, B, C


def point_line_cor(p, A, B, C):
    ##判断点与线之间的位置关系
    # 一般式直线方程(Ax+By+c)=0
    x, y = p
    r = A * x + B * y + C
    return r


def line_to_line(points1, points2, alpha=10, angle=30):
    """
    线段之间的距离
    """
    x1, y1, x2, y2 = points1
    ox1, oy1, ox2, oy2 = points2
    xy = np.array([(x1, y1), (x2, y2)], dtype = 'float32')
    A1, B1, C1 = fit_line(xy)
    oxy = np.array([(ox1, oy1), (ox2, oy2)], dtype = 'float32')
    A2, B2, C2 = fit_line(oxy)
    flag1 = point_line_cor(np.array([x1, y1], dtype = 'float32'), A2, B2, C2)
    flag2 = point_line_cor(np.array([x2, y2], dtype = 'float32'), A2, B2, C2)

    if (flag1 > 0 and flag2 > 0) or (flag1 < 0 and flag2 < 0): # 横线或者竖线在竖线或者横线的同一侧
        if (A1 * B2 - A2 * B1) != 0:
            x = (B1 * C2 - B2 * C1) / (A1 * B2 - A2 * B1)
            y = (A2 * C1 - A1 * C2) / (A1 * B2 - A2 * B1)
            # x, y = round(x, 2), round(y, 2)
            p = (x, y) # 横线与竖线的交点
            r0 = sqrt(p, (x1, y1))
            r1 = sqrt(p, (x2, y2))

            if min(r0, r1) < alpha: # 若交点与线起点或者终点的距离小于alpha，则延长线到交点
                if r0 < r1:
                    k = abs((y2 - p[1]) / (x2 - p[0] + 1e-10))
                    a = atan(k) * 180 / pi
                    if a < angle or abs(90-a) < angle:
                        points1 = np.array([p[0], p[1], x2, y2], dtype = 'float32')
                else:
                    k = abs((y1 - p[1]) / (x1 - p[0] + 1e-10))
                    a = atan(k) * 180 / pi
                    if a < angle or abs(90-a) < angle:
                        points1 = np.array([x1, y1, p[0], p[1]], dtype = 'float32')
    return points1


def final_adjust_lines(rowboxes, colboxes):
    nrow = len(rowboxes)
    ncol = len(colboxes)
    for i in range(nrow):
        for j in range(ncol):
            rowboxes[i] = line_to_line(rowboxes[i], colboxes[j], alpha = 20, angle = 20)
            colboxes[j] = line_to_line(colboxes[j], rowboxes[i], alpha = 20, angle = 20)
    return rowboxes, colboxes


@jit(nopython = True)
def numba_fit_line(p):
    """A = Y2 - Y1
       B = X1 - X2
       C = X2*Y1 - X1*Y2
       AX+BY+C=0
    直线一般方程
    """
    x1, y1 = p[0]
    x2, y2 = p[1]
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return A, B, C

@jit(nopython = True)
def numba_point_line_cor(p, A, B, C):
    ##判断点与线之间的位置关系
    # 一般式直线方程(Ax+By+c)=0
    x, y = p
    r = A * x + B * y + C
    return r

@jit(nopython = True)
def numba_line_to_line(points1, points2, alpha=10, angle=30):
    """
    线段之间的距离
    """
    x1, y1, x2, y2 = points1
    ox1, oy1, ox2, oy2 = points2
    xy = np.array([(x1, y1), (x2, y2)], dtype = 'float32')
    A1, B1, C1 = numba_fit_line(xy)
    oxy = np.array([(ox1, oy1), (ox2, oy2)], dtype = 'float32')
    A2, B2, C2 = numba_fit_line(oxy)
    flag1 = numba_point_line_cor(np.array([x1, y1], dtype = 'float32'), A2, B2, C2)
    flag2 = numba_point_line_cor(np.array([x2, y2], dtype = 'float32'), A2, B2, C2)

    if (flag1 > 0 and flag2 > 0) or (flag1 < 0 and flag2 < 0): # 横线或者竖线在竖线或者横线的同一侧
        if (A1 * B2 - A2 * B1) != 0:
            x = (B1 * C2 - B2 * C1) / (A1 * B2 - A2 * B1)
            y = (A2 * C1 - A1 * C2) / (A1 * B2 - A2 * B1)
            # x, y = round(x, 2), round(y, 2)
            p = (x, y) # 横线与竖线的交点
            r0 = numba_sqrt(p, (x1, y1))
            r1 = numba_sqrt(p, (x2, y2))

            if min(r0, r1) < alpha: # 若交点与线起点或者终点的距离小于alpha，则延长线到交点
                if r0 < r1:
                    k = abs((y2 - p[1]) / (x2 - p[0] + 1e-10))
                    a = atan(k) * 180 / pi
                    if a < angle or abs(90-a) < angle:
                        points1 = np.array([p[0], p[1], x2, y2], dtype = 'float32')
                else:
                    k = abs((y1 - p[1]) / (x1 - p[0] + 1e-10))
                    a = atan(k) * 180 / pi
                    if a < angle or abs(90-a) < angle:
                        points1 = np.array([x1, y1, p[0], p[1]], dtype = 'float32')
    return points1

@jit(nopython = True)
def numba_final_adjust_lines(rowboxes, colboxes):
    nrow = len(rowboxes)
    ncol = len(colboxes)
    for i in range(nrow):
        for j in range(ncol):
            rowboxes[i] = numba_line_to_line(rowboxes[i], colboxes[j], alpha = 20, angle = 30)
            colboxes[j] = numba_line_to_line(colboxes[j], rowboxes[i], alpha = 20, angle = 30)
    return rowboxes, colboxes


def _order_points(pts):
    # 根据x坐标对点进行排序
    # TODO 有问题，bug
    """
    ---------------------
    作者：Tong_T
    来源：CSDN
    原文：https://blog.csdn.net/Tong_T/article/details/81907132
    版权声明：本文为博主原创文章，转载请附上博文链接！
    """
    x_sorted = pts[np.argsort(pts[:, 0]), :]

    left_most = x_sorted[:2, :]
    right_most = x_sorted[2:, :]
    left_most = left_most[np.argsort(left_most[:, 1]), :]
    (tl, bl) = left_most

    distance = dist.cdist(tl[np.newaxis], right_most, "euclidean")[0]
    (br, tr) = right_most[np.argsort(distance)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="float32")


def image_location_sort_box(box):
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    pts = (x1, y1), (x2, y2), (x3, y3), (x4, y4)
    pts = np.array(pts, dtype="float32")
    # (x1, y1), (x2, y2), (x3, y3), (x4, y4) = _order_points(pts)
    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = order_points(pts) #  todo
    return [x1, y1, x2, y2, x3, y3, x4, y4]


def solve(box):
    """
    绕 cx,cy点 w,h 旋转 angle 的坐标
    x = cx-w/2
    y = cy-h/2
    x1-cx = -w/2*cos(angle) +h/2*sin(angle)
    y1 -cy= -w/2*sin(angle) -h/2*cos(angle)

    h(x1-cx) = -wh/2*cos(angle) +hh/2*sin(angle)
    w(y1 -cy)= -ww/2*sin(angle) -hw/2*cos(angle)
    (hh+ww)/2sin(angle) = h(x1-cx)-w(y1 -cy)

    """
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    cx = (x1 + x3 + x2 + x4) / 4.0
    cy = (y1 + y3 + y4 + y2) / 4.0
    w = (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)) / 2
    h = (np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2) + np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2)) / 2
    # x = cx-w/2
    # y = cy-h/2
    sinA = (h * (x1 - cx) - w * (y1 - cy)) * 1.0 / (h * h + w * w) * 2
    angle = np.arcsin(sinA)
    return angle, w, h, cx, cy


def xy_rotate_box(cx, cy, w, h, angle=0, degree=None, **args):
    """
    绕 cx,cy点 w,h 旋转 angle 的坐标
    x_new = (x-cx)*cos(angle) - (y-cy)*sin(angle)+cx
    y_new = (x-cx)*sin(angle) + (y-cy)*sin(angle)+cy
    """
    if degree is not None:
        angle = degree
    cx = float(cx)
    cy = float(cy)
    w = float(w)
    h = float(h)
    angle = float(angle)
    x1, y1 = rotate(cx - w / 2, cy - h / 2, angle, cx, cy)
    x2, y2 = rotate(cx + w / 2, cy - h / 2, angle, cx, cy)
    x3, y3 = rotate(cx + w / 2, cy + h / 2, angle, cx, cy)
    x4, y4 = rotate(cx - w / 2, cy + h / 2, angle, cx, cy)
    return x1, y1, x2, y2, x3, y3, x4, y4


from numpy import cos, sin


def rotate(x, y, angle, cx, cy):
    angle = angle  # *pi/180
    x_new = (x - cx) * cos(angle) - (y - cy) * sin(angle) + cx
    y_new = (x - cx) * sin(angle) + (y - cy) * cos(angle) + cy
    return x_new, y_new

def recalrotateposition(x, y, angle, img, img_new):
    if angle:
        (image_h, image_w) = img.shape[:2]
        cx, cy = image_w / 2, image_h / 2
        new_x, new_y = rotate(x, y, angle * np.pi / 180, cx, cy)
        diffx = (img_new.shape[1] - img.shape[1]) / 2
        diffy = (img_new.shape[0] - img.shape[0]) / 2
        new_x, new_y = new_x + diffx, new_y + diffy
    else:
        new_x, new_y = x, y
    return new_x, new_y

def minAreaRectbox(regions, flag=True, W=0, H=0, filtersmall=False, adjustBox=False):
    """
    多边形外接矩形
    """
    boxes = []
    for region in regions:
        if region.bbox_area > H * W * 3 / 4: # 过滤大的单元格
            continue
        rect = cv2.minAreaRect(region.coords[:, ::-1])

        box = cv2.boxPoints(rect)
        box = box.reshape((8,)).tolist()
        box = image_location_sort_box(box)
        x1, y1, x2, y2, x3, y3, x4, y4 = box
        angle, w, h, cx, cy = solve(box)
        if adjustBox:
            x1, y1, x2, y2, x3, y3, x4, y4 = xy_rotate_box(cx, cy, w + 5, h + 5, angle=0, degree=None)
            x1, x4 = max(x1, 0), max(x4, 0)
            y1, y4 = max(y1, 0), max(y4, 0)

        if w > 32 and h > 32 and flag:
            if abs(angle / np.pi * 180) < 20:
                if filtersmall and (w < 10 or h < 10):
                    continue
                boxes.append([x1, y1, x2, y2, x3, y3, x4, y4])
        else:
            if w * h < 0.5 * W * H:
                if filtersmall and (w < 15 or h < 15):# or w / h > 30 or h / w > 30): # 过滤小的单元格
                    continue
                boxes.append([x1, y1, x2, y2, x3, y3, x4, y4])
    return boxes


from PIL import Image


def rectangle(img, boxes):
    tmp = np.copy(img)
    for box in boxes:
        xmin, ymin, xmax, ymax = box[:4]
        cv2.rectangle(tmp, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 0, 0), 1, lineType=cv2.LINE_AA)
    return Image.fromarray(tmp)


def draw_lines(im, bboxes, color=(0, 0, 0), lineW=3):
    """
        boxes: bounding boxes
    """
    tmp = np.copy(im)
    c = color
    h, w = im.shape[:2]

    for box in bboxes:
        x1, y1, x2, y2 = box[:4]
        cv2.line(tmp, (int(x1), int(y1)), (int(x2), int(y2)), c, lineW, lineType=cv2.LINE_AA)

    return tmp


def draw_boxes(im, bboxes, color=(0, 0, 0)):
    """
        boxes: bounding boxes
    """
    tmp = np.copy(im)
    c = color
    h, w, _ = im.shape
    thick = int((h + w) / 300)
    i = 0
    for box in bboxes:
        if type(box) is dict:
            x1, y1, x2, y2, x3, y3, x4, y4 = xy_rotate_box(**box)
        else:
            x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
        cx  =np.mean([x1,x2,x3,x4])
        cy  = np.mean([y1,y2,y3,y4])
        cv2.line(tmp, (int(x1), int(y1)), (int(x2), int(y2)), c, 1, lineType=cv2.LINE_AA)
        cv2.line(tmp, (int(x2), int(y2)), (int(x3), int(y3)), c, 1, lineType=cv2.LINE_AA)
        cv2.line(tmp, (int(x3), int(y3)), (int(x4), int(y4)), c, 1, lineType=cv2.LINE_AA)
        cv2.line(tmp, (int(x4), int(y4)), (int(x1), int(y1)), c, 1, lineType=cv2.LINE_AA)
        mess = str(i)
        # cv2.putText(tmp, mess, (int(cx), int(cy)), 0, 1e-3 * h / 3 * 2, c, thick * 1 // 3)
        i += 1
    return tmp


def reorganize_box(dt_boxes):
    result_boxes = []
    for i, dt_box in enumerate(dt_boxes):
        dt_box = np.array(dt_box, dtype='float32').reshape(4, 2).tolist()
        x1, y1 = dt_box[0][0], dt_box[0][1]  # left-upper
        x3, y3 = dt_box[2][0], dt_box[2][1]  # right-bottom

        w = x3 - x1
        h = y3 - y1
        center_x = x1 + w // 2
        center_y = y1 + h // 2

        result_box = {'cx': center_x, 'cy': center_y, 'w': w, 'h': h, 'bbox': dt_box}
        result_boxes.append(result_box)
    return result_boxes


def order_rbox(result, alpha=1.5):
    """
    按行合并box
    """

    def diff(box1, box2):
        """
        计算box1,box2之间的距离
        """
        cy1 = box1['cy']
        cy2 = box2['cy']
        h1 = box1['h']
        h2 = box2['h']

        diff_value = abs(cy1 - cy2) / max(0.01, min(h1 / 2, h2 / 2))

        return diff_value

    def sort_group_box(boxes):
        """
        对box进行排序
        """
        N = len(boxes)
        boxes = sorted(boxes, key=lambda x: x['cx'])

        box4 = np.zeros((N, 8))
        for i in range(N):
            bbox = boxes[i]['bbox']
            x1, y1 = bbox[0][0], bbox[0][1]
            x2, y2 = bbox[1][0], bbox[1][1]
            x3, y3 = bbox[2][0], bbox[2][1]
            x4, y4 = bbox[3][0], bbox[3][1]
            box4[i] = [x1, y1, x2, y2, x3, y3, x4, y4]
        return box4.tolist()

    newBox = []
    for line in result:
        if len(newBox) == 0:
            newBox.append([line])
        else:
            check = False
            for box in newBox[-1]:
                if diff(line, box) > alpha:
                    check = True

            if not check:
                newBox[-1].append(line)
            else:
                newBox.append([line])

    newBox = [sort_group_box(bx) for bx in newBox]
    return newBox

def get_crop_image(img, points):
    img_crop_width = int(
        max(
            np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(
            np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],
                          [img_crop_width, img_crop_height],
                          [0, img_crop_height]])
    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC)
    return dst_img


def overlapping_filter(lines, sorting_index, separation=5):
    '''
    horizontal_lines = overlapping_filter(horizontal_lines, 1)
    vertical_lines = overlapping_filter(vertical_lines, 0)
    '''
    filtered_lines = []

    lines = sorted(lines, key=lambda lines: lines[sorting_index])
    # separation = 5
    for i in range(len(lines)):
            l_curr = lines[i]
            if(i>0):
                l_prev = lines[i-1]
                if ( (l_curr[sorting_index] - l_prev[sorting_index]) > separation):
                    filtered_lines.append(l_curr)
            else:
                filtered_lines.append(l_curr)

    return filtered_lines


