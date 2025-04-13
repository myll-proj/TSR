from ocr_system_base import load_model, OCR
from common.params import args
import numpy as np
import cv2

def make_template(Img, final_rboxes):
    # 做模板
    rectangle_dict = {}
    for box in range(len(final_rboxes)):
        x1, y1, x2, y2 = final_rboxes[box]
        boxes = np.array([[x1, y1], [x2, y1], [x2, y2],  [x1, y2] ], dtype=np.int32)
        rectangle_dict[box + 1] = np.int32(boxes).reshape(4,2)

    template = np.zeros(Img.shape[:2], dtype = 'uint16')
    for r in rectangle_dict:
        cv2.fillConvexPoly(template, rectangle_dict[r], r)
    return template, rectangle_dict

class pred_content:
    def __init__(self, image, pred_bbox):
        self.image = image
        self.pred_bbox = pred_bbox

    def result_cell(self):
        img_np = np.array(self.image)

        e2e_algorithm, text_sys = load_model(args, e2e_algorithm = False)
        auto_rotate_whole_image = False
        ocr = OCR(text_sys, img_np, cls = False, char_out = True, e2e_algorithm = e2e_algorithm,auto_rotate_whole_image=auto_rotate_whole_image)
        ocr_result = ocr(union = False, max_x_dist = 1000, min_y_overlap_ratio = 0.5)

        template, rectangle_dict = make_template(img_np, final_rboxes=self.pred_bbox)
        content_boxes_index = list(rectangle_dict.keys())

        ceil_text = {}
        for i in content_boxes_index:
            ceil_text[i] = ""
        ceil_idx = {}
        for i in content_boxes_index:
            ceil_idx[i] = []

        for idx, m in enumerate(ocr_result):
            point = [int(m["bbox"][0] + m["bbox"][2]) // 2, int(m["bbox"][1] + m["bbox"][3]) // 2]  # 中心点
            text = m["text"]
            if point[0]<0 or point[1]<0:
                continue
            try:
                label_ind = template[point[1]][point[0]]
            except:
                label_ind = 0
            if label_ind in content_boxes_index:
                ceil_idx[label_ind].append(idx)
                ceil_text[label_ind] += text + ' '

        pred_cell = []
        for i in ceil_text:
            pred_cell.append(ceil_text[i])

        return pred_cell