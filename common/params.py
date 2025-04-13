# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import platform

import torch


class Base_Config(object):
    def __init__(self, debug=True):
        self.use_gpu = True if torch.cuda.is_available() else False
        if self.use_gpu == True:
            self.device = '0'
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
            self.device = 'cpu'

        self.ir_optim = True
        self.min_subgraph_size = 15
        self.precision = "fp32"
        self.gpu_mem = 500

        self.use_onnx = True #if platform.system().lower() == 'windows' or self.use_gpu == False else False

        # params for text detector
        self.remove_slip = False
        self.slip_angle = 5  # 度数：°
        self.det_algorithm = "DB"
        self.det_model_dir = "inference/ocr/det/det.onnx"
        self.det_limit_side_len = 1696  # 960
        self.det_limit_type = 'max'
        # DB parmas
        self.det_db_thresh = 0.3  # DB模型输出预测图的二值化阈值
        self.det_db_box_thresh = 0.4  # DB模型输出框的阈值，低于此值的预测框会被丢弃
        # self.det_db_unclip_ratio = 2.0  # 检测后处理时控制文本框大小,DB模型输出框扩大的比例
        self.det_db_unclip_ratio = 1.0  # 检测后处理时控制文本框大小,DB模型输出框扩大的比例
        self.use_dilation = True
        self.det_db_score_mode = "fast"

        self.rec_algorithm = "SVTR_LCNet"
        self.rec_image_shape = "3, 48, 320"
        self.rec_model_dir = "inference/ocr/rec/rec.onnx"
        self.rec_char_type = 'ch'
        self.rec_batch_num = 30  # 进行识别时，同时前向的图片数
        self.max_text_length = 25  # 默认训练时的文本可识别的最大长度

        self.rec_char_dict_path = "./ppocr/utils/ppocr_keys_v1.txt"
        self.use_space_char = True
        self.cand_alphabet = None  # todo
        self.vis_font_path = "./ppocr/fonts/simfang.ttf"
        # self.drop_score = 0.5
        self.drop_score = 0.3
        if debug:
            self.is_visualize = True
            self.is_visualize_char = True
        else:
            self.is_visualize = False
            self.is_visualize_char = False
        # self.char_out = True #     是否输出单个字符坐标相关信息
        # self.union = True # 是否水平方向合并检测框
        self.save_crop_res = False
        self.crop_res_save_dir = r'./test/output'
        self.save_path = './test'

        # params for text classifier
        self.use_angle_cls = False
        self.cls_model_dir = ""
        self.cls_image_shape = "3, 48, 192"
        self.label_list = ['0', '180']
        self.cls_batch_num = 30
        self.cls_thresh = 0.9

        self.enable_mkldnn = False
        self.cpu_threads = 10
        self.use_pdserving = False
        self.warmup = False
        self.use_tensorrt = False

        self.use_mp = True
        self.total_process_num = 4
        self.process_id = 0
        self.ir_optim = True
        self.benchmark = False
        self.show_log = True

        self.layout_model_dir = r'inference/table_detect'
        self.layout_dict_path = r'ppocr/utils/dict/layout_dict/layout_table_dict.txt'
        self.layout_score_threshold = 0.5
        self.layout_nms_threshold = 0.5
        self.layout_size = [800, 608]  # h, w

        if debug:
            self.isToExcel = True
            self.table_show = True
            self.table_save = True
        else:
            self.isToExcel = False
            self.table_show = False
            self.table_save = False

        self.is_slide = False
        self.is_resize = True
        self.config = r'table_structure_recognition/models/erfnet/erfnet_fcn_4x4_1024x1024_160k_common_table.py'
        self.checkpoint = r'table_structure_recognition/models/erfnet/best_mIoU_epoch_297.pth'
        self.opacity = 0.5  # Opacity of painted segmentation map. In (0, 1] range
        self.out_file = 'test/mmseg_result.jpg'  # None


    def __getattr__(self, item):
        return item

debug = True
args = Base_Config(debug = debug)



if __name__ == '__main__':
    print(args.__dict__)
    print(args.use_gpu)