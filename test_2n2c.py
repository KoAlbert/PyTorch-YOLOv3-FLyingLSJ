from __future__ import division

from models import *
from utils.utils import *

from utils.datasets import *
from utils.parse_config import *

import os
import sys
import time
import datetime
import argparse
import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
from PIL import Image




def evaluate_2n2c(model, valid_path, iou_thres, conf_thres, nms_thres, img_size, batch_size, detection_result_folder, cachedir, class_names, color_class_num, obj_class_num):
    model.eval()

    # Get dataloader
    dataset = ListDataset_2n2c(valid_path, img_size=img_size, augment=False, multiscale=False, class_num = color_class_num + obj_class_num)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    img_path_list = []  # Stores image paths
    img_detections_list = []  # Stores detections for each image index

    for batch_i, (img_path, input_imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):

        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        input_imgs = Variable(input_imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():            
            detections = model(input_imgs)

            detections = non_max_suppression_2n2c(detections, conf_thres, nms_thres, color_class_num, obj_class_num)

            #print('------after--------detections={}'.format(detections))
            
            img_path_list.extend(img_path)
            img_detections_list.extend(detections)


    for img_i, (img_path, detections) in enumerate(zip(img_path_list, img_detections_list)):

        if detections is not None:

            img = np.array(Image.open(img_path)) # for reading image size to rescale x,y,w,h back
            detections = rescale_boxes(detections, img_size, img.shape[:2])        

            boxes_writing_multi_label(detections, detection_result_folder, class_names , img_path, conf_thres, color_class_num, obj_class_num)

    APs = do_python_eval_quite_multi_label(detection_result_folder, valid_path, cachedir, class_names, iou_thres)


    return APs


#python test_2n2c.py

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch: can only be 1 in this example")
    parser.add_argument("--model_def", type=str, default="config/yolov3-custom_2n2c.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/custom_2n2c.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/2n2c/yolov3_ckpt_5.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/custom_2n2c/classes.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.01, help="object confidence threshold; for evaluation, it could be 0.01")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--detection model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--color_class_num", type=int, default=2, help="the number of color classes")
    parser.add_argument("--obj_class_num", type=int, default=2, help="the number of obj classes")
    parser.add_argument("--detection_result_folder", type=str, default="./detection_result_folder/2n2c/", help="folder to store detection txt files")
    parser.add_argument("--cachedir", type=str, default="./annotations_cache/2n2c/", help="folder to store cache files of labels")

        
    opt = parser.parse_args()
    #print(opt)

    ### remove cache files under cache folder
    files = glob.glob(opt.cachedir + '*')

    for f in files:
        os.remove(f)
    ###    

    ### remove files under detection_result_dir
    files = glob.glob(opt.detection_result_folder + '*')

    for f in files:
        os.remove(f)
    ###

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])


    assert len(class_names) == (opt.color_class_num + opt.obj_class_num)

    # Initiate model
    model = Darknet(opt.model_def).to(device)
    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    print("Compute mAP...")
    

    APs = evaluate_2n2c(
        model,
        valid_path=valid_path,
        iou_thres=opt.iou_thres,
        conf_thres=opt.conf_thres,
        nms_thres=opt.nms_thres,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
        detection_result_folder = opt.detection_result_folder,
        cachedir = opt.cachedir,
        class_names = class_names,
        color_class_num=opt.color_class_num,
        obj_class_num=opt.obj_class_num,
    )

    print(APs)
    mAP = np.array(list(APs.values())).mean()


    print(mAP)


