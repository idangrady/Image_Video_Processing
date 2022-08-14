import numpy as np
import torch
from IOt import *


def nms(bboxs, iou_trsh,prob_tresh, box_format = "corners"):
    # class, prob, x1,y1,x2,y2
    best_objects = {}
    bboxs = [box for box in bboxs if box[1]>prob_tresh]
    bboxs = sorted(bboxs, key= lambda x: x[1], reverse=True)
    boxes_after_nms = []
    while bboxs:
        cur_box = bboxs.pop(0)
        cur_class_, cur_prob, cur_x1, cur_y1, cur_x2, cur_y2 = cur_box
        bboxs = [
            box
            for box in bboxs
            if box[0] != cur_box[0] # of there are not at the same class, we dont want to compare them
            or IOver_U(torch.tensor(cur_box[2:]),torch.tensor(box[2:]),box_format=box_format) < iou_trsh # should check is
        ]
        boxes_after_nms.append(cur_box)

    return boxes_after_nms


