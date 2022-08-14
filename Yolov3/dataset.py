import numpy as np
import os
import pandas as pd
import torch
from PIL import Image,ImageFile
from torch.utils.data import DataLoader,Dataset

from ..IOt import *
from ..nms import *



ImageFile.LOAD_TRUNCATED_IMAGES = True

class YOLOdataset(Dataset):
    def __init__(self,csv_file, img_dir, label_dir, anchors, image_size = 416, class_size = 20, transform = None):
        self.annotation = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.image_size = image_size
        self.class_size = class_size

        self.transform = transform

        self.anchors = torch.tensor(anchors[0]+ anchors[1]+ anchors[2]) # put all input together with different scales
        self.nun_anchors = self.anchor.shape[0]
        self.num_anchors_per_schalr = self.num_anchors//3

        self.ignore_iou_tresh = 0.5

    def __len__(self):
        return len(self.annotation)
    def __getitem__(self, idx):
        label_path = os.path.join(self.label_dir, self.annotation.iloc[idx, 1])
        bbx = np.roll(np.loadtxt(frame = label_path, dlimiter = " ", ndim = 2), 4, axis = 1).tolist() # from (class, x,y,h,w) to (x,y,h,w,class)
        img_path = os.path.join(self.img_dir, self.annotation.iloc[idx, 0])

        img = np.array(Image.open(img_path).convert("RGB"))

        if(self.transform):
            """:cvar
            Transform the image based on the func
            """
            augmentation = self.transform(image = img, bboxex = bbx) # we do augmentation, and preserve the bounding boxex
            image = augmentation["image"]
        else:
            pass

        return self.annotation[idx]

