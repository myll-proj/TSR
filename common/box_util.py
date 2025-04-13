# # Copyright (c) OpenMMLab. All rights reserved.
import numpy as np

def coord_convert(bboxes):
    # 4 points coord to 2 points coord for rectangle bbox
    x_min, y_min, x_max, y_max = \
        min(bboxes[0::2]), min(bboxes[1::2]), max(bboxes[0::2]), max(bboxes[1::2])
    return [x_min, y_min, x_max, y_max]

def is_on_same_line(box_a, box_b, min_y_overlap_ratio=0.8):
    """Check if two boxes are on the same line by their y-axis coordinates.

    Two boxes are on the same line if they overlap vertically, and the length
    of the overlapping line segment is greater than min_y_overlap_ratio * the
    height of either of the boxes.

    Args:
        box_a (list), box_b (list): Two bounding boxes to be checked
        min_y_overlap_ratio (float): The minimum vertical overlapping ratio
                                    allowed for boxes in the same line

    Returns:
        The bool flag indicating if they are on the same line
    """
    a_y_min = np.min(box_a[1::2])
    b_y_min = np.min(box_b[1::2])
    a_y_max = np.max(box_a[1::2])
    b_y_max = np.max(box_b[1::2])

    if a_y_min >= b_y_min and a_y_max <= b_y_max:  # todo
        return True

    if b_y_min >= a_y_min and b_y_max <= a_y_max:  # todo
        return True

    # Make sure that box a is always the box above another
    if a_y_min > b_y_min:
        a_y_min, b_y_min = b_y_min, a_y_min
        a_y_max, b_y_max = b_y_max, a_y_max

    # if b_y_min <= a_y_max and np.min(box_a[0::2]) != np.min(box_b[0::2]): # todo 新增and np.min(box_a[0::2]) != np.min(box_b[0::2])
    if b_y_min <= a_y_max and box_a[0] != box_b[0]:  # todo 新增and box_a[0] != box_b[0]
        if min_y_overlap_ratio is not None:
            sorted_y = sorted([b_y_min, b_y_max, a_y_max])
            overlap = sorted_y[1] - sorted_y[0]
            min_a_overlap = (a_y_max - a_y_min) * min_y_overlap_ratio
            min_b_overlap = (b_y_max - b_y_min) * min_y_overlap_ratio
            return overlap >= min_a_overlap or \
                overlap >= min_b_overlap
        else:
            return True
    return False


def stitch_boxes_into_lines(boxes, max_x_dist=10, min_y_overlap_ratio=0.8):
    """Stitch fragmented boxes of words into lines.

    Note: part of its logic is inspired by @Johndirr
    (https://github.com/faustomorales/keras-ocr/issues/22)

    Args:
        boxes (list): List of ocr results to be stitched
        max_x_dist (int): The maximum horizontal distance between the closest
                    edges of neighboring boxes in the same line
        min_y_overlap_ratio (float): The minimum vertical overlapping ratio
                    allowed for any pairs of neighboring boxes in the same line

    Returns:
        merged_boxes(list[dict]): List of merged boxes and texts
    """

    if len(boxes) <= 1:
        return boxes

    merged_boxes = []

    # sort groups based on the x_min coordinate of boxes
    x_sorted_boxes = sorted(boxes, key=lambda x: np.min(x['box'][::2]))
    # store indexes of boxes which are already parts of other lines
    skip_idxs = set()

    i = 0
    # locate lines of boxes starting from the leftmost one
    for i in range(len(x_sorted_boxes)):
        if i in skip_idxs:
            continue
        # the rightmost box in the current line
        rightmost_box_idx = i
        line = [rightmost_box_idx]
        for j in range(i + 1, len(x_sorted_boxes)):
            if j in skip_idxs:
                continue
            if is_on_same_line(x_sorted_boxes[rightmost_box_idx]['box'],
                               x_sorted_boxes[j]['box'], min_y_overlap_ratio):
                line.append(j)
                skip_idxs.add(j)
                rightmost_box_idx = j

        # split line into lines if the distance between two neighboring
        # sub-lines' is greater than max_x_dist
        lines = []
        line_idx = 0
        lines.append([line[0]])
        for k in range(1, len(line)):
            curr_box = x_sorted_boxes[line[k]]
            prev_box = x_sorted_boxes[line[k - 1]]
            dist = np.min(curr_box['box'][::2]) - np.max(prev_box['box'][::2])
            if dist > max_x_dist:
                line_idx += 1
                lines.append([])
            lines[line_idx].append(line[k])

        # Get merged boxes
        for box_group in lines:
            merged_box = {}
            merged_box['text'] = ' '.join([x_sorted_boxes[idx]['text'] for idx in box_group])

            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = float('-inf'), float('-inf')
            origin_box = []
            score_lst = []
            for idx in box_group:
                score_lst.append(x_sorted_boxes[idx]['score'])
                origin_box.append(x_sorted_boxes[idx]['box'])
                x_max = max(np.max(x_sorted_boxes[idx]['box'][::2]), x_max)
                x_min = min(np.min(x_sorted_boxes[idx]['box'][::2]), x_min)
                y_max = max(np.max(x_sorted_boxes[idx]['box'][1::2]), y_max)
                y_min = min(np.min(x_sorted_boxes[idx]['box'][1::2]), y_min)

            merged_box['quadrangle'] = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
            merged_box['box'] = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
            merged_box['origin_box'] = origin_box

            charslist = []
            for idx in box_group:
                charslist.extend(x_sorted_boxes[idx].get('chars', []))
            if charslist != []:
                merged_box['chars'] = charslist
            merged_box['score'] = np.mean(score_lst)

            merged_boxes.append(merged_box)

    bboxes = [i['box'] for i in merged_boxes] # Todo
    bboxes = np.array(bboxes)
    bboxes = sorted_boxes(bboxes)

    index_dict = {f'{k}': i for i, k in enumerate(bboxes)}
    merged_boxes = sorted(merged_boxes, key = lambda x: index_dict[f"{x['box']}"])

    # bboxes_sorted = [None] * len(bboxes)
    # for bg_item in merged_boxes:
    #     idx = bboxes.index(bg_item['box'])
    #     bboxes_sorted[idx] = bg_item

    return merged_boxes


def stitch_boxes_into_lines_v2(boxes, max_x_dist=10, min_y_overlap_ratio=0.8, symbol=' '):
    """Stitch fragmented boxes of words into lines.

    Note: part of its logic is inspired by @Johndirr
    (https://github.com/faustomorales/keras-ocr/issues/22)

    Args:
        boxes (list): List of ocr results to be stitched
        max_x_dist (int): The maximum horizontal distance between the closest
                    edges of neighboring boxes in the same line
        min_y_overlap_ratio (float): The minimum vertical overlapping ratio
                    allowed for any pairs of neighboring boxes in the same line

    Returns:
        merged_boxes(list[dict]): List of merged boxes and texts
    """

    if len(boxes) <= 1:
        return boxes

    merged_boxes = []

    # sort groups based on the x_min coordinate of boxes
    x_sorted_boxes = sorted(boxes, key=lambda x: np.min(x['box'][::2]))
    # store indexes of boxes which are already parts of other lines
    skip_idxs = set()

    # i = 0
    # locate lines of boxes starting from the leftmost one
    for i in range(len(x_sorted_boxes)):
        bbox = x_sorted_boxes[i]['bbox']
        h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
        if h / w * 1.0 > 1.8: # todo, 竖直文本不合并
            x_sorted_boxes[i].update({'origin_box': [x_sorted_boxes[i]['box']]})  # todo 解决缺失origi_box的bug 20221103
            merged_boxes.append(x_sorted_boxes[i])
            skip_idxs.add(i)
            continue
        if i in skip_idxs:
            continue
        # the rightmost box in the current line
        rightmost_box_idx = i
        line = [rightmost_box_idx]
        for j in range(i + 1, len(x_sorted_boxes)):
            bbox = x_sorted_boxes[j]['bbox']
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h / w * 1.0 > 1.8:  # todo, 竖直文本不合并
                skip_idxs.add(j)
                continue
            if j in skip_idxs:
                continue
            if is_on_same_line(x_sorted_boxes[rightmost_box_idx]['box'],
                               x_sorted_boxes[j]['box'], min_y_overlap_ratio):
                line.append(j)
                skip_idxs.add(j)
                rightmost_box_idx = j

        # split line into lines if the distance between two neighboring
        # sub-lines' is greater than max_x_dist
        lines = []
        line_idx = 0
        lines.append([line[0]])
        for k in range(1, len(line)):
            curr_box = x_sorted_boxes[line[k]]
            prev_box = x_sorted_boxes[line[k - 1]]
            dist = np.min(curr_box['box'][::2]) - np.max(prev_box['box'][::2])
            if dist > max_x_dist:
                line_idx += 1
                lines.append([])
            lines[line_idx].append(line[k])

        # Get merged boxes
        for box_group in lines:
            merged_box = {}
            merged_box['text'] = symbol.join([x_sorted_boxes[idx]['text'] for idx in box_group])

            x_min, y_min = float('inf'), float('inf')
            x_max, y_max = float('-inf'), float('-inf')
            origin_box = []
            score_lst = []
            for idx in box_group:
                score_lst.append(x_sorted_boxes[idx]['score'])
                origin_box.append(x_sorted_boxes[idx]['box'])
                x_max = max(np.max(x_sorted_boxes[idx]['box'][::2]), x_max)
                x_min = min(np.min(x_sorted_boxes[idx]['box'][::2]), x_min)
                y_max = max(np.max(x_sorted_boxes[idx]['box'][1::2]), y_max)
                y_min = min(np.min(x_sorted_boxes[idx]['box'][1::2]), y_min)

            merged_box['quadrangle'] = [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
            merged_box['box'] = [x_min, y_min, x_max, y_min, x_max, y_max, x_min, y_max]
            merged_box['origin_box'] = origin_box

            charslist = []
            for idx in box_group:
                charslist.extend(x_sorted_boxes[idx].get('chars', []))
            if charslist != []:
                merged_box['chars'] = charslist
            merged_box['score'] = np.mean(score_lst)

            merged_boxes.append(merged_box)

    bboxes = [i['box'] for i in merged_boxes] # Todo
    bboxes = np.array(bboxes)
    end2end_sorted_idx_list, end2end_sorted_bbox_list, sorted_groups, sorted_bbox_groups = sort_bbox(bboxes, min_y_overlap_ratio)

    # index_dict = {f'{k}': i for i, k in enumerate(end2end_sorted_bbox_list)}
    # new_merged_boxes = sorted(merged_boxes, key = lambda x: index_dict[f"{x['box']}"])

    new_merged_boxes = [None] * len(end2end_sorted_bbox_list)
    for bg_item in merged_boxes:
        idx = end2end_sorted_bbox_list.index(bg_item['box'])
        new_merged_boxes[idx] = bg_item

    return new_merged_boxes


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[1], x[0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i + 1][1] - _boxes[i][1]) < 10 and \
                (_boxes[i + 1][0] < _boxes[i][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    _boxes = [i.tolist() for i in _boxes]
    return _boxes



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
    # todo 修改，避免sorted_xs中x值相同
    xs = [bg_item[0] for bg_item in bg]
    xs = [str(index) + '_' + str(i) for index, i in enumerate(xs)]
    sorted_xs = sorted(xs, key = lambda x: float(x.split('_')[1]))

    g_sorted = [None]*len(xs)
    bg_sorted = [None]*len(xs)
    for index, (g_item, bg_item) in enumerate(zip(g, bg)):
        # idx = xs_sorted.index(bg_item[0])
        idx = sorted_xs.index(str(index) + '_' + str(bg_item[0]))
        bg_sorted[idx] = bg_item
        g_sorted[idx] = g_item

    return g_sorted, bg_sorted


def flatten(sorted_groups, sorted_bbox_groups):
    idxs = []
    bboxes = []
    for group, bbox_group in zip(sorted_groups, sorted_bbox_groups):
        for g, bg in zip(group, bbox_group):
            idxs.append(g)
            bboxes.append(bg.tolist())
    return idxs, bboxes


def sort_bbox(end2end_xyxy_bboxes, min_y_overlap_ratio = 0.2):
    """
    This function will group the render end2end bboxes in row.
    :param end2end_xyxy_bboxes:
    :return:
    """
    groups = []
    bbox_groups = []
    for index, end2end_xywh_bbox in enumerate(end2end_xyxy_bboxes):
        this_bbox = end2end_xywh_bbox
        temp = coord_convert(this_bbox)
        w, h = temp[2] - temp[0], temp[3] - temp[1]
        if len(groups)==0 or h / w > 1.8: #  todo
            groups.append([index])
            bbox_groups.append([this_bbox])
        else:
            flag = False
            for g, bg in zip(groups, bbox_groups):
                temp = coord_convert(bg[0]) # todo
                w, h = temp[2] - temp[0], temp[3] - temp[1]
                if h / w > 1.8:
                    continue
                # this_bbox is belong to bg's row or not
                # if is_abs_lower_than_threshold(this_bbox, bg[0], threshold=threshold):
                if is_on_same_line(this_bbox, bg[0], min_y_overlap_ratio = min_y_overlap_ratio):
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
    # sorted_groups = [None]*len(tmp_groups)
    # sorted_bbox_groups = [None]*len(tmp_bbox_groups)
    # ys = [bg[0][1] for bg in tmp_bbox_groups]
    # sorted_ys = sorted(ys)
    # for g, bg in zip(tmp_groups, tmp_bbox_groups):
    #     idx = sorted_ys.index(bg[0][1])
    #     sorted_groups[idx] = g
    #     sorted_bbox_groups[idx] = bg

    # todo 修改，避免sorted_ys中y值相同
    sorted_groups = [None]*len(tmp_groups)
    sorted_bbox_groups = [None]*len(tmp_bbox_groups)
    ys = [bg[0][1] for bg in tmp_bbox_groups]
    ys = [str(index) + '_' + str(i) for index, i in enumerate(ys)]
    sorted_ys = sorted(ys, key = lambda x: float(x.split('_')[1]))
    for index, (g, bg) in enumerate(zip(tmp_groups, tmp_bbox_groups)):
        # idx = sorted_ys.index(bg[0][1])
        idx = sorted_ys.index(str(index) + '_'+ str(bg[0][1]))
        sorted_groups[idx] = g
        sorted_bbox_groups[idx] = bg

    # flatten, get final result
    end2end_sorted_idx_list, end2end_sorted_bbox_list \
        = flatten(sorted_groups, sorted_bbox_groups)

    # check sorted
    #img = cv2.imread('/data_0/yejiaquan/data/TableRecognization/singleVal/PMC3286376_004_00.png')
    #img = drawBboxAfterSorted(img, sorted_groups, sorted_bbox_groups)

    return end2end_sorted_idx_list, end2end_sorted_bbox_list, sorted_groups, sorted_bbox_groups


def fourxy2eightxy(dt_boxes):
    new_dt_boxes = dt_boxes.reshape(dt_boxes.shape[0], -1)
    return new_dt_boxes

def eightxy2fourxy(dt_boxes):
    new_dt_boxes = dt_boxes.reshape(dt_boxes.shape[0], -1, 2)
    return new_dt_boxes



