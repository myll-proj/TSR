import os
import re
import cv2
import glob
import copy
import math
import pickle
import numpy as np

from tqdm import tqdm
from common.match_utils import xywh2xyxy, xyxy2xywh, remove_empty_bboxes
from shapely.geometry import Polygon,MultiPoint
from collections import OrderedDict

"""
Useful function in matching.
"""


def convert_coord(xyxy):
    """
    Convert two points format to four points format.
    :param xyxy:
    :return:
    """
    new_bbox = np.zeros([4,2], dtype=np.float32)
    new_bbox[0,0], new_bbox[0,1] = xyxy[0], xyxy[1]
    new_bbox[1,0], new_bbox[1,1] = xyxy[2], xyxy[1]
    new_bbox[2,0], new_bbox[2,1] = xyxy[2], xyxy[3]
    new_bbox[3,0], new_bbox[3,1] = xyxy[0], xyxy[3]
    return new_bbox


def cal_iou(bbox1, bbox2):
    bbox1_poly = Polygon(bbox1).convex_hull
    bbox2_poly = Polygon(bbox2).convex_hull
    union_poly = np.concatenate((bbox1, bbox2))

    if not bbox1_poly.intersects(bbox2_poly):
        iou = 0
    else:
        inter_area = bbox1_poly.intersection(bbox2_poly).area
        union_area = MultiPoint(union_poly).convex_hull.area
        if union_area == 0:
            iou = 0
        else:
            iou = float(inter_area) / union_area
    return iou


def cal_distance(p1, p2):
    delta_x = p1[0] - p2[0]
    delta_y = p1[1] - p2[1]
    d = math.sqrt((delta_x ** 2) + (delta_y ** 2))
    return d


def is_inside(center_point, corner_point):
    """
    Find if center_point inside the bbox(corner_point) or not.
    :param center_point: center point (x, y)
    :param corner_point: corner point ((x1,y1),(x2,y2))
    :return:
    """
    x_flag = False
    y_flag = False
    if (center_point[0] >= corner_point[0][0]) and (center_point[0] <= corner_point[1][0]):
        x_flag = True
    if (center_point[1] >= corner_point[0][1]) and (center_point[1] <= corner_point[1][1]):
        y_flag = True
    if x_flag and y_flag:
        return True
    else:
        return False


def find_no_match(match_list, all_end2end_nums, type='end2end'):
    """
    Find out no match end2end bbox in previous match list.
    :param match_list: matching pairs.
    :param all_end2end_nums: numbers of end2end_xywh
    :param type: 'end2end' corresponding to idx 0, 'master' corresponding to idx 1.
    :return: no match pse bbox index list
    """
    if type == 'end2end':
        idx = 0
    elif type == 'master':
        idx = 1
    else:
        raise ValueError

    no_match_indexs = []
    # m[0] is end2end index m[1] is master index
    matched_bbox_indexs = [m[idx] for m in match_list]
    for n in range(all_end2end_nums):
        if n not in matched_bbox_indexs:
            no_match_indexs.append(n)
    return no_match_indexs


def is_abs_lower_than_threshold(this_bbox, target_bbox, threshold=3):
    # only consider y axis, for grouping in row.
    delta = abs(this_bbox[1] - target_bbox[1])
    if delta < threshold:
        return True
    else:
        return False


def sort_line_bbox(g, bg):
    """
    Sorted the bbox in the same line(group)
    compare coord 'x' value, where 'y' value is closed in the same group.
    :param g: index in the same group
    :param bg: bbox in the same group
    :return:
    """

    xs = [bg_item[0] for bg_item in bg]
    xs_sorted = sorted(xs)

    g_sorted = [None]*len(xs_sorted)
    bg_sorted = [None]*len(xs_sorted)
    for g_item, bg_item in zip(g, bg):
        idx = xs_sorted.index(bg_item[0])
        bg_sorted[idx] = bg_item
        g_sorted[idx] = g_item

    return g_sorted, bg_sorted


def flatten(sorted_groups, sorted_bbox_groups):
    idxs = []
    bboxes = []
    for group, bbox_group in zip(sorted_groups, sorted_bbox_groups):
        for g, bg in zip(group, bbox_group):
            idxs.append(g)
            bboxes.append(bg)
    return idxs, bboxes


def sort_bbox(end2end_xywh_bboxes, no_match_end2end_indexes):
    """
    This function will group the render end2end bboxes in row.
    :param end2end_xywh_bboxes:
    :param no_match_end2end_indexes:
    :return:
    """
    groups = []
    bbox_groups = []
    for index, end2end_xywh_bbox in zip(no_match_end2end_indexes, end2end_xywh_bboxes):
        this_bbox = end2end_xywh_bbox
        if len(groups)==0:
            groups.append([index])
            bbox_groups.append([this_bbox])
        else:
            flag = False
            for g, bg in zip(groups, bbox_groups):
                # this_bbox is belong to bg's row or not
                if is_abs_lower_than_threshold(this_bbox, bg[0]):
                    g.append(index)
                    bg.append(this_bbox)
                    flag = True
                    break
            if not flag:
                # this_bbox is not belong to bg's row, create a row.
                groups.append([index])
                bbox_groups.append([this_bbox])

    # sorted bboxes in a group
    tmp_groups, tmp_bbox_groups = [], []
    for g, bg in zip(groups, bbox_groups):
        g_sorted, bg_sorted = sort_line_bbox(g, bg)
        tmp_groups.append(g_sorted)
        tmp_bbox_groups.append(bg_sorted)

    # sorted groups, sort by coord y's value.
    sorted_groups = [None]*len(tmp_groups)
    sorted_bbox_groups = [None]*len(tmp_bbox_groups)
    ys = [bg[0][1] for bg in tmp_bbox_groups]
    sorted_ys = sorted(ys)
    for g, bg in zip(tmp_groups, tmp_bbox_groups):
        idx = sorted_ys.index(bg[0][1])
        sorted_groups[idx] = g
        sorted_bbox_groups[idx] = bg

    # flatten, get final result
    end2end_sorted_idx_list, end2end_sorted_bbox_list \
        = flatten(sorted_groups, sorted_bbox_groups)

    # check sorted
    #img = cv2.imread('/data_0/yejiaquan/data/TableRecognization/singleVal/PMC3286376_004_00.png')
    #img = drawBboxAfterSorted(img, sorted_groups, sorted_bbox_groups)

    return end2end_sorted_idx_list, end2end_sorted_bbox_list, sorted_groups, sorted_bbox_groups


def get_bboxes_list(end2end_result, structure_master_result):
    """
    This function is use to convert end2end results and structure master results to
    List of xyxy bbox format and List of xywh bbox format
    :param end2end_result: bbox's format is xyxy
    :param structure_master_result: bbox's format is xywh
    :return: 4 kind list of bbox ()
    """
    # end2end
    end2end_xyxy_list = []
    end2end_xywh_list = []
    for end2end_item in end2end_result:
        src_bbox = np.array(end2end_item['bbox'])
        end2end_xyxy_list.append(src_bbox)
        xywh_bbox = xyxy2xywh(src_bbox)
        end2end_xywh_list.append(xywh_bbox)
    end2end_xyxy_bboxes = np.array(end2end_xyxy_list)
    end2end_xywh_bboxes = np.array(end2end_xywh_list)

    # structure master
    src_bboxes = structure_master_result['bbox']
    # src_bboxes = remove_empty_bboxes(src_bboxes)
    structure_master_xyxy_bboxes = src_bboxes
    xywh_bboxes = xyxy2xywh(src_bboxes)
    structure_master_xywh_bboxes = xywh_bboxes

    return end2end_xyxy_bboxes, end2end_xywh_bboxes, structure_master_xywh_bboxes, structure_master_xyxy_bboxes


def center_rule_match(end2end_xywh_bboxes, structure_master_xyxy_bboxes):
    """
    Judge end2end Bbox's center point is inside structure master Bbox or not,
    if end2end Bbox's center is in structure master Bbox, get matching pair.
    :param end2end_xywh_bboxes:
    :param structure_master_xyxy_bboxes:
    :return: match pairs list, e.g. [[0,1], [1,2], ...]
    """
    match_pairs_list = []
    for i, end2end_xywh in enumerate(end2end_xywh_bboxes):
        for j, master_xyxy in enumerate(structure_master_xyxy_bboxes):
            x_end2end, y_end2end = end2end_xywh[0], end2end_xywh[1]
            x_master1, y_master1, x_master2, y_master2 \
                = master_xyxy[0], master_xyxy[1], master_xyxy[2], master_xyxy[3]
            center_point_end2end = (x_end2end, y_end2end)
            corner_point_master = ((x_master1, y_master1), (x_master2, y_master2))
            if is_inside(center_point_end2end, corner_point_master):
                match_pairs_list.append([i, j])
    return match_pairs_list


def iou_rule_match(end2end_xyxy_bboxes, end2end_xyxy_indexes, structure_master_xyxy_bboxes):
    """
    Use iou to find matching list.
    choose max iou value bbox as match pair.
    :param end2end_xyxy_bboxes:
    :param end2end_xyxy_indexes: original end2end indexes.
    :param structure_master_xyxy_bboxes:
    :return: match pairs list, e.g. [[0,1], [1,2], ...]
    """
    match_pair_list = []
    for end2end_xyxy_index, end2end_xyxy in zip(end2end_xyxy_indexes, end2end_xyxy_bboxes):
        max_iou = 0
        max_match = [None, None]
        for j, master_xyxy in enumerate(structure_master_xyxy_bboxes):
            end2end_4xy = convert_coord(end2end_xyxy)
            master_4xy = convert_coord(master_xyxy)
            iou = cal_iou(end2end_4xy, master_4xy)
            if iou > max_iou:
                max_match[0], max_match[1] = end2end_xyxy_index, j
                max_iou = iou

        if max_match[0] is None:
            # no match
            continue
        match_pair_list.append(max_match)
    return match_pair_list


# def distance_rule_match(end2end_indexes, end2end_bboxes, master_indexes, master_bboxes):
#     """
#     Get matching between no-match end2end bboxes and no-match master bboxes.
#     Use min distance to match.
#     This rule will only run (no-match end2end nums > 0) and (no-match master nums > 0)
#     It will Return master_bboxes_nums match-pairs.
#     :param end2end_indexes:
#     :param end2end_bboxes:
#     :param master_indexes:
#     :param master_bboxes:
#     :return: match_pairs list, e.g. [[0,1], [1,2], ...]
#     """
#     min_match_list = []
#     for j, master_bbox in zip(master_indexes, master_bboxes):
#         min_distance = np.inf
#         min_match = [0, 0]  # i, j
#         for i, end2end_bbox in zip(end2end_indexes, end2end_bboxes):
#             x_end2end, y_end2end = end2end_bbox[0], end2end_bbox[1]
#             x_master, y_master = master_bbox[0], master_bbox[1]
#             end2end_point = (x_end2end, y_end2end)
#             master_point = (x_master, y_master)
#             dist = cal_distance(master_point, end2end_point)
#             if dist < min_distance:
#                 min_match[0], min_match[1] = i, j
#                 min_distance = dist
#         min_match_list.append(min_match)
#     return min_match_list

def distance_rule_match(end2end_indexes, end2end_bboxes, master_indexes, master_bboxes):
    """
    Get matching between no-match end2end bboxes and no-match master bboxes.
    Use min distance to match.
    This rule will only run (no-match end2end nums > 0) and (no-match master nums > 0)
    It will Return master_bboxes_nums match-pairs.
    :param end2end_indexes:
    :param end2end_bboxes:
    :param master_indexes:
    :param master_bboxes:
    :return: match_pairs list, e.g. [[0,1], [1,2], ...]
    """
    min_match_list = []
    for i, master_bbox in zip(end2end_indexes, end2end_bboxes):
        min_distance = np.inf
        min_match = [0, 0]  # i, j
        for j, end2end_bbox in zip(master_indexes, master_bboxes):
            x_end2end, y_end2end = end2end_bbox[0], end2end_bbox[1]
            x_master, y_master = master_bbox[0], master_bbox[1]
            end2end_point = (x_end2end, y_end2end)
            master_point = (x_master, y_master)
            dist = cal_distance(master_point, end2end_point)
            if dist < min_distance:
                min_match[0], min_match[1] = i, j
                min_distance = dist
        min_match_list.append(min_match)
    return min_match_list

def extra_match(no_match_end2end_indexes, master_bbox_nums):
    """
    This function will create some virtual master bboxes,
    and get match with the no match end2end indexes.
    :param no_match_end2end_indexes:
    :param master_bbox_nums:
    :return:
    """
    end_nums = len(no_match_end2end_indexes) + master_bbox_nums
    extra_match_list = []
    for i in range(master_bbox_nums, end_nums):
        end2end_index = no_match_end2end_indexes[i-master_bbox_nums]
        extra_match_list.append([end2end_index, i])
    return extra_match_list


def get_match_dict(match_list):
    """
    Convert match_list to a dict, where key is master bbox's index, value is end2end bbox index.
    :param match_list:
    :return:
    """
    match_dict = dict()
    for match_pair in match_list:
        end2end_index, master_index = match_pair[0], match_pair[1]
        if master_index not in match_dict.keys():
            match_dict[master_index] = [end2end_index]
        else:
            match_dict[master_index].append(end2end_index)
    return match_dict


def get_match_text_dict(match_dict, end2end_info, break_token=''):
    match_text_dict = dict()
    for master_index, end2end_index_list in match_dict.items():
        text_list = [end2end_info[end2end_index]['text'] for end2end_index in end2end_index_list]
        text = break_token.join(text_list)
        match_text_dict[master_index] = text
    return match_text_dict


class Matcher:
    def __init__(self, end2end_result, structure_master_result):
        """
        This class process the end2end results and structure recognition results.
        :param end2end_results: end2end results predict by end2end inference.
        :param structure_master_results: structure recognition results predict by structure master inference.
        """
        self.end2end_result = end2end_result
        self.structure_master_result = structure_master_result

    def match(self):
        """
        Match process:
        pre-process : convert end2end and structure master results to xyxy, xywh ndnarray format.
        1. Use pseBbox is inside masterBbox judge rule
        2. Use iou between pseBbox and masterBbox rule
        3. Use min distance of center point rule
        :return:
        """
        end2end_result = self.end2end_result
        structure_master_result = self.structure_master_result

        match_list = []
        end2end_xyxy_bboxes, end2end_xywh_bboxes, structure_master_xywh_bboxes, structure_master_xyxy_bboxes = \
            get_bboxes_list(end2end_result, structure_master_result)

        # rule 1: center rule
        center_rule_match_list = \
            center_rule_match(end2end_xywh_bboxes, structure_master_xyxy_bboxes)
        match_list.extend(center_rule_match_list)

        # rule 2: iou rule
        # firstly, find not match index in previous step.
        # center_no_match_end2end_indexs = \
        #     find_no_match(match_list, len(end2end_xywh_bboxes), type='end2end')
        # if len(center_no_match_end2end_indexs) > 0:
        #     center_no_match_end2end_xyxy = end2end_xyxy_bboxes[center_no_match_end2end_indexs]
        #     # secondly, iou rule match
        #     iou_rule_match_list = \
        #         iou_rule_match(center_no_match_end2end_xyxy, center_no_match_end2end_indexs, structure_master_xyxy_bboxes)
        #     match_list.extend(iou_rule_match_list)

        return match_list


    def get_merge_result(self, match_results):
        """
        Merge the OCR result into structure token to get final results.
        :param match_results:
        :return:
        """

        # break_token is linefeed token, when one master bbox has multiply end2end bboxes.
        break_token = ' '

        end2end_info = self.end2end_result
        match_dict = get_match_dict(match_results)
        match_text_dict = get_match_text_dict(match_dict, end2end_info, break_token)
        return match_text_dict