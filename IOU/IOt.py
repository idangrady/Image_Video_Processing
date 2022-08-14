import torch


def IOver_U(boxes_pred, box_labels, box_format = "Midpoint"):

    if box_format =='Midpoint':
        b1_x1 = (boxes_pred[..., 0:1] - boxes_pred[..., 2:3])/2
        b1_y1 = (boxes_pred[..., 1:2] - boxes_pred[..., 3:4])/2
        b1_x2 = (boxes_pred[..., 0:1] + boxes_pred[..., 2:3])/2
        b1_y2 =(boxes_pred[..., 1:2] + boxes_pred[..., 3:4])/2


        b2_x1 = (box_labels[..., 0:1] - box_labels[..., 2:3])/2
        b2_y1 = (box_labels[..., 1:2] - box_labels[..., 3:4])/2
        b2_x2 = (box_labels[..., 0:1] + box_labels[..., 2:3])/2
        b2_y2 =(box_labels[..., 1:2] + box_labels[..., 3:4])/2


    if box_format =='corners':
        b1_x1 = boxes_pred[..., 0:1]
        b1_y1 = boxes_pred[...,1:2]
        b1_x2 = boxes_pred[..., 2:3]
        b1_y2 = boxes_pred[...,3:4]

        b2_x1 = box_labels[..., 0:1]
        #Boxes Pred Shape
        b2_y1 = box_labels[...,1:2]
        b2_x2 = box_labels[..., 2:3]
        b2_y2 = box_labels[...,3:4]

    x_1 = torch.max(b1_x1, b2_x1)
    y_2 = torch.max(b1_y1, b2_y1)

    x_2 = torch.min(b1_x2, b2_x2)
    y_1 = torch.min(b1_y2, b2_y2)
    # for the case where there is no intersection
    intersection = (x_2-x_1).clamp(0) * (y_2 -y_1).clamp(0)

    b_1_area = abs((b1_x2- b1_x1)* (b1_y2-b1_y1))
    b_2_area = abs((b2_x2- b2_x1)* (b2_y2-b2_y1))

    return (intersection/(b_1_area + b_2_area - intersection))

