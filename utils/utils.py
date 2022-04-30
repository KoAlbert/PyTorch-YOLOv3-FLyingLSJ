from __future__ import division
import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pickle
import os
from PIL import Image

def to_cpu(tensor):
    return tensor.detach().cpu()


def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    """
    对权重进行初始化
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape

    #print('-----------orig_h={}'.format(orig_h))
    #print('-----------orig_w={}'.format(orig_w))


    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 4]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    #print('------b1_x1.shape={}'.format(b1_x1.shape))
    #print('------b2_x1.shape={}'.format(b2_x1.shape))

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
    
    return iou


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output

def non_max_suppression_2n2c(prediction, conf_thres=0.5, nms_thres=0.4, color_class_num = 2, obj_class_num=2):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:(5+obj_class_num + color_class_num)].max(1)[0]
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        # class_confs: top cofidence values of each bounding boxes
        # class_preds: the class (amoung B, R, Sedan, SUV) of highest confidence
        #class_confs, class_preds = image_pred[:, 5:(5+color_class_num)].max(1, keepdim=True) # the class used to perform NMS; only color in this case

        # the class used to perform NMS; only vehicle type in this case; but the last class type meaning will be different
        color_class_confs, color_class_preds = image_pred[:, 5:(5+color_class_num)].max(1, keepdim=True)
        car_class_confs, car_class_preds = image_pred[:, (5+color_class_num):(5+color_class_num+obj_class_num)].max(1, keepdim=True)

        #detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        # 
        detections = torch.cat((image_pred[:, :5], image_pred[:, 5:(5+color_class_num + obj_class_num)], color_class_preds.float(), car_class_preds.float()), 1)

        #print('-----color_class_confs={}'.format(color_class_confs))
        #print('-----color_class_preds={}'.format(color_class_preds))
        #print('-----car_class_confs={}'.format(car_class_confs))
        #print('-----car_class_preds={}'.format(car_class_preds))
        #print('-----detections={}'.format(detections))
        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    return output

def NMS_revised(prediction, conf_thres=0.5, nms_thres=0.4):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    #print('------prediction.shape={}'.format(prediction.shape))
    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        image_pred = image_pred[image_pred[:, 4] >= conf_thres]
        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Object confidence times class confidence
        #score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]
        score = image_pred[:, 4] * image_pred[:, 5:6].max(1)[0] # only sorting by car-class; image_pred[:, 4]:objectness; image_pred[:, 5:6]:car class probability
        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        # class_confs: max confidence
        # class_preds: index of max confidence
        #class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        class_confs, class_preds = image_pred[:, 5:7].max(1, keepdim=True) # NMS is performed only by car-class

        class_preds[class_preds==1] = 11
        #print('----------class_confs={}'.format(class_confs))
        #print('----------class_preds={}'.format(class_preds))

        #detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        detections = torch.cat((image_pred[:, :5], image_pred[:, 5:].float(), class_preds.float()), 1)


        # Perform non-maximum suppression
        keep_boxes = []
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            weights = detections[invalid, 4:5]
            # Merge overlapping bboxes by order of confidence
            detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)

    #print('----------output={}'.format(output))
    return output

def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)

    #print('---------------nC={}'.format(nC))

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()

    """
    #print('======target_labels.shape={}'.format(target_labels.shape))
    target_labels1 = torch.zeros_like(target_labels) # target_labels: nx1
    target_labels2 = torch.zeros_like(target_labels)

    target_labels1[target_labels==0] = 0
    target_labels2[target_labels==0] = 0

    target_labels1[target_labels==1] = 1
    target_labels2[target_labels==1] = 1

    target_labels1[target_labels==11] = 0
    target_labels2[target_labels==11] = 1
    """

    #print('---------------target_labels.shape={}'.format(target_labels.shape))
    #print('---------------target_labels={}'.format(target_labels))
    #print('---------------target_labels1={}'.format(target_labels1))
    #print('---------------target_labels2={}'.format(target_labels2))
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1 # original

    #tcls[b, best_n, gj, gi, target_labels1] = 1
    #tcls[b, best_n, gj, gi, target_labels2] = 1

    #print('=======tcls[b, best_n, gj, gi, target_labels1]={}'.format(tcls[b, best_n, gj, gi, target_labels1]))
    #print('=======tcls[b, best_n, gj, gi, target_labels2]={}'.format(tcls[b, best_n, gj, gi, target_labels2]))

    # Compute label correctness and iou at best anchor
    #print('=======pred_cls[b, best_n, gj, gi].shape={}'.format(pred_cls[b, best_n, gj, gi].shape))
    # class_mask & iou_scores are designed for estimating class accuracy and iou scores, respectively.
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    #print('======class_mask[b, best_n, gj, gi]={}'.format(class_mask[b, best_n, gj, gi]))
    #print('======class_mask.shape={}'.format(class_mask.shape))
    #print('======obj_mask.shape={}'.format(obj_mask.shape))

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf


def build_targets_2n1c(pred_boxes, pred_cls, target, anchors, ignore_thres):

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)

    print('---------------nB={}'.format(nB))
    print('---------------nA={}'.format(nA))
    print('---------------nG={}'.format(nG))
    print('---------------nC={}'.format(nC))
    #print('---------------nC={}'.format(nC))

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()

    #print('======target_labels.shape={}'.format(target_labels.shape))
    target_labels1 = torch.zeros_like(target_labels) # target_labels: nx1
    target_labels2 = torch.zeros_like(target_labels)

    target_labels1[target_labels==0] = 0
    target_labels2[target_labels==0] = 0

    target_labels1[target_labels==1] = 1
    target_labels2[target_labels==1] = 1

    target_labels1[target_labels==11] = 0
    target_labels2[target_labels==11] = 1

    print('---------------target_labels.shape={}'.format(target_labels.shape))
    print('---------------target_labels={}'.format(target_labels))
    print('---------------target_labels1={}'.format(target_labels1))
    print('---------------target_labels2={}'.format(target_labels2))
    print('---------------b={}'.format(b))


    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    #tcls[b, best_n, gj, gi, target_labels] = 1 # original

    tcls[b, best_n, gj, gi, target_labels1] = 1
    tcls[b, best_n, gj, gi, target_labels2] = 1

    #print('=======tcls[b, best_n, gj, gi, target_labels1]={}'.format(tcls[b, best_n, gj, gi, target_labels1]))
    #print('=======tcls[b, best_n, gj, gi, target_labels2]={}'.format(tcls[b, best_n, gj, gi, target_labels2]))

    #print('=======tcls[b, best_n, gj, gi, target_labels1].shape={}'.format(tcls[b, best_n, gj, gi, target_labels1].shape))
    #print('=======tcls[b, best_n, gj, gi, target_labels2].shape={}'.format(tcls[b, best_n, gj, gi, target_labels2].shape))    

    print('=======tcls[b, best_n, gj, gi, :]={}'.format(tcls[b, best_n, gj, gi, :]))


    #print('=======b[0]={}'.format(b[0]))
    #print('=======b[0].shape={}'.format(b[0].shape))

    #print('=======tcls[b, best_n, gj, gi, :].shape={}'.format(tcls[b, best_n, gj, gi, :].shape))
    #print('=======tcls[b[0], best_n, gj, gi, :].shape={}'.format(tcls[b[0], best_n, gj, gi, :].shape))
    #print('=======tcls[0, best_n, gj, gi, :].shape={}'.format(tcls[0, best_n, gj, gi, :].shape))

    # Compute label correctness and iou at best anchor
    #print('=======pred_cls[b, best_n, gj, gi].shape={}'.format(pred_cls[b, best_n, gj, gi].shape))
    # class_mask & iou_scores are designed for estimating class accuracy and iou scores, respectively.
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    #print('======class_mask[b, best_n, gj, gi]={}'.format(class_mask[b, best_n, gj, gi]))
    #print('======class_mask.shape={}'.format(class_mask.shape))
    #print('======obj_mask.shape={}'.format(obj_mask.shape))

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf


def build_targets_2n2c(pred_boxes, pred_cls, target, anchors, ignore_thres, num_classes):

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)

    #print('---------------nB={}'.format(nB))
    #print('---------------nA={}'.format(nA))
    #print('---------------nG={}'.format(nG))
    #print('---------------nC={}'.format(nC))

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    #print('---------------tcls.shape={}'.format(tcls.shape))

    # Convert to position relative to box
#    target_boxes = target[:, 2:6] * nG
#    gxy = target_boxes[:, :2]
#    gwh = target_boxes[:, 2:]

    target_boxes = target[:, (num_classes+1):(num_classes+5)] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]


    
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    # Separate target values
    #b, target_labels = target[:, :2].long().t()
    b, target_label_dummy = target[:, :2].long().t()

    #print('---------------b={}'.format(b))
    #print('---------------target_label_dummy.size()[0]={}'.format(target_label_dummy.size()[0]))
    #print('---------------target[:, :2].shape={}'.format(target[:, :2].shape))
    #print('---------------target[:, :2].t().shape={}'.format(target[:, :2].t().shape))

    """
    target_label_bit = torch.zeros_like(target_label_dummy)

    for i in range(0,num_classes):
        target_label_bit = target[:, (1+i):(2+i)].long().t()



    print('======target_labels.shape={}'.format(target_labels.shape))
    target_labels1 = torch.zeros_like(target_labels) # target_labels: nx1
    target_labels2 = torch.zeros_like(target_labels)

    target_labels1[target_labels==0] = 0
    target_labels2[target_labels==0] = 0

    target_labels1[target_labels==1] = 1
    target_labels2[target_labels==1] = 1

    target_labels1[target_labels==11] = 0
    target_labels2[target_labels==11] = 1
    """
    #print('---------------target_labels.shape={}'.format(target_labels.shape))
    #print('---------------target_labels={}'.format(target_labels))
    #print('---------------target_labels1={}'.format(target_labels1))
    #print('---------------target_labels2={}'.format(target_labels2))
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    #tcls[b, best_n, gj, gi, target_labels] = 1 # original

    #tcls[b, best_n, gj, gi, target_labels1] = 1
    #tcls[b, best_n, gj, gi, target_labels2] = 1

    for num_class in range(0, num_classes):               
        target_label_bit = torch.ones_like(target_label_dummy)*num_class
        #print('---------------target_label_bit.shape={}'.format(target_label_bit.shape))
        #print('---------------target_label_bit={}'.format(target_label_bit))
        tcls[b, best_n, gj, gi, target_label_bit] = target[:, (1+num_class)]
        #print('---------------tcls[b, best_n, gj, gi, target_label_bit].shape={}'.format(tcls[b, best_n, gj, gi, target_label_bit].shape))
        #print('---------------target[:, (1+num_class)].shape={}'.format(target[:, (1+num_class)].shape))
        #print('---------------target[:, (1+num_class)]={}'.format(target[:, (1+num_class)]))

    #print(tcls[b, best_n, gj, gi, :])
    """
    for row_index in range(0,target_label_dummy.size()[0]):                
        #print('--------------- tcls[b[row_index], best_n, gj, gi, :].shape={}'.format( tcls[b[row_index], best_n, gj, gi, :].shape))
        print('---------------row_index={}'.format(row_index))
        print('---------------b[row_index]={}'.format(b[row_index]))
        print('---------------tcls[b[row_index], best_n, gj, gi, :].shape={}'.format(tcls[b[row_index], best_n, gj, gi, :].shape))
        print('---------------target[row_index, 1:(1+num_classes)].long().t().shape={}'.format(target[row_index, 1:(1+num_classes)].long().t().shape))
        print('---------------')
        tcls[b[row_index], best_n, gj, gi, :] = target[row_index, 1:(1+num_classes)].t()   
    """
    #print('=======tcls[b, best_n, gj, gi, target_labels1]={}'.format(tcls[b, best_n, gj, gi, target_labels1]))
    #print('=======tcls[b, best_n, gj, gi, target_labels2]={}'.format(tcls[b, best_n, gj, gi, target_labels2]))

    # Compute label correctness and iou at best anchor
    #print('=======pred_cls[b, best_n, gj, gi].shape={}'.format(pred_cls[b, best_n, gj, gi].shape))
    # class_mask & iou_scores are designed for estimating class accuracy and iou scores, respectively.
    #class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    #iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    #print('======class_mask[b, best_n, gj, gi]={}'.format(class_mask[b, best_n, gj, gi]))
    #print('======class_mask.shape={}'.format(class_mask.shape))
    #print('======obj_mask.shape={}'.format(obj_mask.shape))

    tconf = obj_mask.float()
    #return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf
    return obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf


def parse_rec_multi_class(txt_filename, img_file_path, classnames):
    
    assert os.path.exists(txt_filename) == True
    assert os.path.exists(img_file_path) == True

    with open(txt_filename, 'r') as f:
        lines = f.readlines()

    img = np.array(Image.open(img_file_path))
    img_h, img_w = img.shape[:2]

    objects = []

    num_class = len(classnames) 
    #print(class_names)
    #print(classnames)

    for line in lines:

        #for index, classname in enumerate(classnames):
        line_ele = line.strip().split(' ')
        #print('---------line_ele={}'.format(line_ele))
        #if int(line_ele[index]) == 1:
        center_x = int(float(line_ele[1])*img_w)
        center_y = int(float(line_ele[2])*img_h)
        w = int(float(line_ele[3])*img_w)
        h = int(float(line_ele[4])*img_h)

        xmin = center_x - (w/2)
        ymin = center_y - (h/2)

        xmax = center_x + (w/2)
        ymax = center_y + (h/2)

        xmin = np.maximum(xmin,0)
        ymin = np.maximum(ymin,0)

        xmax = np.minimum(xmax,img_w)
        ymax = np.minimum(ymax,img_h)

        obj_struct = {}
        obj_struct['name'] = classnames[int(line_ele[0])]
        obj_struct['bbox'] = [xmin, ymin, xmax, ymax]
        obj_struct['truncated'] = int(0)
        obj_struct['difficult'] = int(0)
        objects.append(obj_struct)

    #print('---objects={}'.format(objects))
    return objects


def parse_rec_multi_label(txt_filename, img_file_path, classnames):
    
    assert os.path.exists(txt_filename) == True
    assert os.path.exists(img_file_path) == True

    with open(txt_filename, 'r') as f:
        lines = f.readlines()

    img = np.array(Image.open(img_file_path))
    img_h, img_w = img.shape[:2]

    objects = []

    num_class = len(classnames) 
    #print(class_names)
    #print(classnames)

    for line in lines:

        for index, classname in enumerate(classnames):
            line_ele = line.strip().split(' ')
            #print('---------line_ele={}'.format(line_ele))
            if int(line_ele[index]) == 1:

                center_x = int(float(line_ele[num_class+0])*img_w)
                center_y = int(float(line_ele[num_class+1])*img_h)
                w = int(float(line_ele[num_class+2])*img_w)
                h = int(float(line_ele[num_class+3])*img_h)

                xmin = center_x - (w/2)
                ymin = center_y - (h/2)

                xmax = center_x + (w/2)
                ymax = center_y + (h/2)

                xmin = np.maximum(xmin,0)
                ymin = np.maximum(ymin,0)

                xmax = np.minimum(xmax,img_w)
                ymax = np.minimum(ymax,img_h)

                obj_struct = {}
                obj_struct['name'] = classname
                obj_struct['bbox'] = [xmin, ymin, xmax, ymax]
                obj_struct['truncated'] = int(0)
                obj_struct['difficult'] = int(0)
                objects.append(obj_struct)

    #print('---objects={}'.format(objects))
    return objects



def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval_multi_class(detpath,                          
             imagesetfile,
             classname,
             classnames,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval_multi_label(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    #imagenames = [x.strip() for x in lines]
    ## there is a \n counted a single character so [0:-5] instead of [0:-4]
    imagenames = [x.split('/')[-1].split('.')[0] for x in lines]
    txt_full_path = [ line.strip().replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt") for line in lines ]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec_multi_class(txt_full_path[i], lines[i].strip(), classnames )
            
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    #print(classname)
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    #print(class_recs)
    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
                    #print("there is a fp 1")
        else:
            fp[d] = 1.

    
    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap


def voc_eval_multi_label(detpath,                          
             imagesetfile,
             classname,
             classnames,
             cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval_multi_label(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])

    Top level function that does the PASCAL VOC evaluation.

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    if not os.path.isdir(cachedir):
        os.mkdir(cachedir)
    cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    #imagenames = [x.strip() for x in lines]
    ## there is a \n counted a single character so [0:-5] instead of [0:-4]
    imagenames = [x.split('/')[-1].split('.')[0] for x in lines]
    txt_full_path = [ line.strip().replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt") for line in lines ]

    if not os.path.isfile(cachefile):
        # load annots
        recs = {}
        for i, imagename in enumerate(imagenames):
            recs[imagename] = parse_rec_multi_label(txt_full_path[i], lines[i].strip(), classnames )
            
            if i % 100 == 0:
                print('Reading annotation for {:d}/{:d}'.format(
                    i + 1, len(imagenames)))
        # save
        print('Saving cached annotations to {:s}'.format(cachefile))
        with open(cachefile, 'wb') as f:
            pickle.dump(recs, f)
    else:
        # load
        with open(cachefile, 'rb') as f:
            recs = pickle.load(f)

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    #print(classname)
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    #print(class_recs)
    # read dets
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        if BBGT.size > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(BBGT[:, 0], bb[0])
            iymin = np.maximum(BBGT[:, 1], bb[1])
            ixmax = np.minimum(BBGT[:, 2], bb[2])
            iymax = np.minimum(BBGT[:, 3], bb[3])
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
                   (BBGT[:, 2] - BBGT[:, 0] + 1.) *
                   (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
                    #print("there is a fp 1")
        else:
            fp[d] = 1.

    
    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap

def boxes_writing(detections, detection_result_folder, class_names , file_path, conf_thres):        
    
    imgfilename_no_ext  = file_path.split('/')[-1].split('.')[0]

    #print(detections)
#   detection_result_txt_blue, detection_result_txt_red, detection_result_txt_sedan, detection_result_txt_SUV
    #for i, detection_result_txt in enumerate(class_names):
          

    #for x1, y1, x2, y2, conf, cls_conf_blue, cls_conf_red, cls_conf_sedan, cls_conf_SUV, color_cls_pred, car_cls_pred in detections:
    for detection in detections:
        try:
            with open(os.path.join(detection_result_folder,class_names[int(detection[-1])] + '.txt'), "a+") as file: # or just open      
                if detection[4]*detection[5] >= conf_thres:
                    x1 = detection[0]
                    y1 = detection[1]
                    x2 = detection[2]
                    y2 = detection[3]

                    file.write('%s %.3f %.1f %.1f %.1f %.1f\n' %
                            (imgfilename_no_ext, detection[4]*detection[5], x1 + 1, y1 + 1, x2 + 1, y2 + 1))

        except IOError:
            # raise error or print
            print('--the file is is not opened sucessfully')

            
def boxes_writing_multi_label(detections, detection_result_folder, class_names , file_path, conf_thres, color_class_num, obj_class_num):        

    assert len(class_names) == color_class_num + obj_class_num
    imgfilename_no_ext  = file_path.split('/')[-1].split('.')[0]

#   detection_result_txt_blue, detection_result_txt_red, detection_result_txt_sedan, detection_result_txt_SUV
    for i, detection_result_txt in enumerate(class_names):
        try:
            with open(os.path.join(detection_result_folder,detection_result_txt + '.txt'), "a+") as file: # or just open                

                #for x1, y1, x2, y2, conf, cls_conf_blue, cls_conf_red, cls_conf_sedan, cls_conf_SUV, color_cls_pred, car_cls_pred in detections:
                for detection in detections:
                    if detection[4]*detection[5+i] >= conf_thres:
                        x1 = detection[0]
                        y1 = detection[1]
                        x2 = detection[2]
                        y2 = detection[3]

                        file.write('%s %.3f %.1f %.1f %.1f %.1f\n' %
                                (imgfilename_no_ext, detection[5+i], x1 + 1, y1 + 1, x2 + 1, y2 + 1))
            
        except IOError:
            # raise error or print
            print('--the file is is not opened sucessfully')

def do_python_eval_quite_multi_label(detection_result_dir, imagesetfile, cachedir, classnames, IoU_thresh):

    # filename = '/data/hongji/darknet/results/comp4_det_test_{:s}.txt'
    filename = os.path.join(detection_result_dir, '{:s}.txt')

    result = dict()
    # The PASCAL VOC metric changed in 2010
    use_07_metric = False # 2007 is not used anymore

    for i, classname in enumerate(classnames):

        detection_file = filename.format(classname)
        #print(detection_file)
        
        if os.path.exists(detection_file):            
            rec, prec, ap = voc_eval_multi_label(detection_file, imagesetfile, classname, classnames, cachedir, ovthresh=IoU_thresh, use_07_metric=use_07_metric)
        else:
            ap = 0.0
                
        result[classname] = ap                        

    result_without_nan = {a1:b1 if math.isnan(b1) == False else 0.0 for a1, b1 in result.items()}
    result_without_nan = np.nan_to_num(result_without_nan)
    
    mAP = np.array(list(result_without_nan.values())).mean()

    return result_without_nan    

def do_python_eval_quite_multi_class(detection_result_dir, imagesetfile, cachedir, classnames, IoU_thresh):

    # filename = '/data/hongji/darknet/results/comp4_det_test_{:s}.txt'
    filename = os.path.join(detection_result_dir, '{:s}.txt')

    result = dict()
    # The PASCAL VOC metric changed in 2010
    use_07_metric = False # 2007 is not used anymore

    for i, classname in enumerate(classnames):

        detection_file = filename.format(classname)
        #print(detection_file)
        
        if os.path.exists(detection_file):            
            rec, prec, ap = voc_eval_multi_class(detection_file, imagesetfile, classname, classnames, cachedir, ovthresh=IoU_thresh, use_07_metric=use_07_metric)
        else:
            ap = 0.0
                
        result[classname] = ap                        

    result_without_nan = {a1:b1 if math.isnan(b1) == False else 0.0 for a1, b1 in result.items()}
    result_without_nan = np.nan_to_num(result_without_nan)
    
    mAP = np.array(list(result_without_nan.values())).mean()

    return result_without_nan    