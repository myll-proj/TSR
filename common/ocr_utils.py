import re
import time
from PIL import Image
from scipy.signal import find_peaks
from scipy.ndimage import filters, interpolation
from numpy import amin, amax
import numpy as np
import base64
from io import BytesIO
import cv2
from loguru import logger as log
from pandas import Series
from common.params import Base_Config
args = Base_Config()

def resize_im(im, scale, max_scale=None, return_f=False):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    if return_f:
        if f >=1:
            return cv2.resize(im, (0, 0), fx=f, fy=f, interpolation = cv2.INTER_CUBIC), f
        else:
            return cv2.resize(im, (0, 0), fx=f, fy=f, interpolation = cv2.INTER_AREA), f
    else:
        if f >=1:
            return cv2.resize(im, (0, 0), fx=f, fy=f, interpolation = cv2.INTER_CUBIC)
        else:
            return cv2.resize(im, (0, 0), fx = f, fy = f)

def estimate_skew_angle(raw, angleRange=[-15, 15]):
    """
    估计图像文字偏转角度,
    计算行的方差，最大的方差就是对应偏移角度
    angleRange:角度估计区间
    """
    scale = max(raw.shape[:2])
    raw = Image.fromarray(raw)
    raw = np.array(raw.convert('L'))
    raw = resize_im(raw, scale=600, max_scale=900)
    # raw = resize_im(raw, scale=scale)
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
    # t0 = time.time()
    # estimates = []
    # for a in angles:
    #     roest = interpolation.rotate(est, a, order=0, mode='constant')
    #     v = np.mean(roest, axis=1)
    #     v = np.var(v)
    #     estimates.append((v, a))
    roests = [interpolation.rotate(est, a, order=0, mode='constant') for a in angles]
    vs = [np.var(np.mean(roest, axis = 1)) for roest in roests]
    estimates = zip(vs, list(angles))
    # print(time.time()-t0)
    _, a = max(estimates)
    return a

def eval_angle(img, degree):
    """
    估计图片文字的偏移角度
    """
    im = Image.fromarray(img)
    # degree = estimate_skew_angle(np.array(im.convert('L')), angleRange=angleRange)
    im = im.rotate(degree, center=(im.size[0] / 2, im.size[1] / 2), expand=1, fillcolor=(255, 255, 255))
    img = np.array(im)
    return img

def removered(image):
    B_channel,G_channel,R_channel=cv2.split(image)
    B_channel = B_channel * 0.6
    G_channel = G_channel * 0.4
    img_array = cv2.subtract(R_channel, G_channel.astype(np.uint8))
    img_array = cv2.subtract(img_array, B_channel.astype(np.uint8))
    img_array = cv2.add(R_channel, img_array)
    img_array = cv2.bilateralFilter(img_array, 2, 32 ,30)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    return img_array

# def removered(image):
#     B_channel,G_channel,R_channel=cv2.split(image)
#     hist = cv2.calcHist([R_channel], [0], None, [256], [0, 256])
#     hist = np.append(hist, [0])
#     hist_squ = hist.squeeze()
#     peaks, properties = find_peaks(hist_squ, height = hist_squ.max() // 5, width = 2)
#     if len(peaks) == 0:
#         median = 162
#     elif 0 < len(peaks) <=2:
#         median = peaks[0]
#     else:
#         median = peaks[len(peaks)//2]
#     value = int(median * 0.618)
#     _, img_array = cv2.threshold(R_channel, value, 255, cv2.THRESH_BINARY)
#     img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
#     cv2.imwrite('./test/adjust.jpg', img_array)
#     import matplotlib.pyplot as plt
#     plt.plot(hist)
#     plt.show()
#     return img_array

def cal_center(box_lst):
    '''
    box: [121.0, 25.0, 206.0, 25.0, 206.0, 53.0, 121.0, 53.0]
    '''
    center_lst = []
    for box in box_lst:
        x1, y1 = box[0], box[1]  # left-upper
        x3, y3 = box[4], box[5]  # right-bottom
        w = x3 - x1
        h = y3 - y1
        center_x = x1 + w // 2
        center_y = y1 + h // 2
        center_lst.append([center_x, center_y])
    return center_lst


def reorganize_box(dt_boxes, rec_res):
    result_boxes = []
    for i, dt_box in enumerate(dt_boxes):
        x1, y1 = dt_box[0][0], dt_box[0][1] # left-upper
        x3, y3 = dt_box[2][0], dt_box[2][1] # right-bottom
        text = rec_res[i][0]
        score = rec_res[i][1]
        chars_list = rec_res[i][2]

        w = x3 - x1
        h = y3 - y1
        center_x = x1 + w // 2
        center_y = y1 + h // 2

        result_box = {'text': text, 'score': score, 'cx': center_x, 'cy':center_y,\
                    'w': w, 'h': h, 'bbox': dt_box, 'degree': 0, 'chars_list': chars_list}

        result_boxes.append(result_box)
    # print(result_boxes)
    return result_boxes

# def string_to_arrimg(img_str):
#     '''
#     读取base64字符，转化为矩阵数据
#     '''
#     try:
#         if isinstance(img_str, str):
#             img_str = img_str.encode('utf-8')
#         img_data = base64.b64decode(img_str)
#         buffer = BytesIO(img_data)
#         img = Image.open(buffer).convert('RGB')
#         img_arr = np.array(img)
#         log.info('image shape is:%s'%str(img_arr.shape))
#         return img_arr
#     except:
#         return None

def string_to_arrimg(img_str, log_flag=False):
    '''
    读取base64字符，保存图像
    '''
    try:
        if isinstance(img_str, str):
            img_str = img_str.encode('utf-8')
        img_data = base64.b64decode(img_str)

        img_arr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 不能转为RGB，否则会出错
        if log_flag:
            log.info('image shape is:%s'%str(img.shape))
        return img
    except:
        return None

def arrimg_to_string(img):
    '''
    图像数据转化为base64
    '''
    img = Image.fromarray(img)
    buffer = BytesIO()
    img.save(buffer, format='png')
    img = buffer.getvalue()
    img_str = base64.b64encode(img).decode('utf8')
    return img_str

def arrimg2string(img):
    img = cv2.imencode('.jpg', img)[1]
    base64_data = base64.b64encode(img).decode('utf8')
    return base64_data

def imagefile_to_string(filename):
   with open(filename,"rb") as f:#转为二进制格式
       img_str = base64.b64encode(f.read()).decode('utf8')#使用base64进行加密
       #print(base64_data)
   return img_str

def get_sub_img(img, scale):
    '''
    抠图
    '''
    m, n, _ = img.shape
    x_min = int(min(scale[:, 0]))
    x_max = int(max(scale[:, 0]))
    y_min = int(min(scale[:, 1]))
    y_max = int(max(scale[:, 1]))

    sub_img = img[max(0, y_min): min(m, y_max), max(0, x_min): min(n, x_max):]
    return sub_img


def get_rotate_crop_image(img, points):
    '''
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    '''
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
    # dst_img_height, dst_img_width = dst_img.shape[0:2]
    # if dst_img_height * 1.0 / dst_img_width >= 1.5:
    #     dst_img = np.rot90(dst_img)
    return dst_img


## 图片旋转
def rotate_bound(image, angle):
    # 获取宽高
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # 提取旋转矩阵 sin cos
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    # cos = np.abs(M[0, 0])
    # sin = np.abs(M[0, 1])

    # # 计算图像的新边界尺寸
    # nW = int((h * sin) + (w * cos))
    # nH = int((h * cos) + (w * sin))
    # #nH = h

    # # 调整旋转矩阵
    # M[0, 2] += (nW / 2) - cX
    # M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


## 获取图片旋转角度
def get_minAreaRect(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    return cv2.minAreaRect(coords)

def order_points(pts):
    rect = np.zeros((4, 2), dtype='float32')
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def fourxy2twoxy(quadrangle):
    quadrangle = np.array(quadrangle)
    pot_lf = float(min(quadrangle[:, 0]))
    pot_tp = float(min(quadrangle[:, 1]))
    pot_rt = float(max(quadrangle[:, 0]))
    pot_bm = float(max(quadrangle[:, 1]))
    bbox = [pot_lf, pot_tp, pot_rt, pot_bm]
    return bbox

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
    box = box.reshape(1, -1).squeeze().tolist()
    x1, y1, x2, y2, x3, y3, x4, y4 = box
    cx = (x1 + x3 + x2 + x4) / 4.0
    cy = (y1 + y3 + y4 + y2) / 4.0
    w = (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)) / 2
    h = (np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2) + np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2)) / 2
    # x = cx-w/2
    # y = cy-h/2

    sinA = (h * (x1 - cx) - w * (y1 - cy)) * 1.0 / (h * h + w * w) * 2
    if abs(sinA) > 1:
        angle = None
    else:
        angle = np.arcsin(sinA)
    return angle, w, h, cx, cy

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # dst = np.array([
    #     [0, 0],
    #     [maxWidth - 1, 0],
    #     [maxWidth - 1, maxHeight - 1],
    #     [0, maxHeight - 1]], dtype="float32")
    dst = np.array([
        [50, 50],
        [maxWidth - 50, 50],
        [maxWidth - 50, maxHeight - 50],
        [50, maxHeight - 50]], dtype = "float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped

def re_map(regulation, text):
    for i in regulation:
        res = re.search(i, text)
        if res:
            return res


