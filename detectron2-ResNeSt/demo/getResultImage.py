# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import math
import numpy as np
import platform
import numba
from numba import jit

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo

# constants
WINDOW_NAME = "COCO detections"


def allfiles(path):
    res = []
    
    for root, dirs, files in os.walk(path):
        for file in files:
            filepath = os.path.join(root, file)
            res.append(filepath)

    return res

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    #cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.001
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.3
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def getOverlapAreaSize(coor1, coor2) :
    x1_new = max(coor1[0], coor2[0]).cpu().numpy()
    y1_new = max(coor1[1], coor2[1]).cpu().numpy()
    x2_new = min(coor1[2], coor2[2]).cpu().numpy()
    y2_new = min(coor1[3], coor2[3]).cpu().numpy()

    if x2_new - x1_new >= 0 and y2_new - y1_new >= 0 :
        return ((x2_new - x1_new) * (y2_new - y1_new)).astype(np.int)
    else :
        return 0

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        if len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        
        for file in allfiles(args.input[0]):
            # use PIL, to be consistent with evaluation
            img = read_image(file, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            pred_masks_nuc = []
            pred_masks_cyp = []
            pred_boxes_nuc = []
            pred_boxes_cyp = []
            for i in range(len(predictions["instances"])) :
                result = (predictions["instances"][i])
                if result.pred_classes == 0:
                    pred_masks_cyp.append(result.pred_masks)
                    pred_boxes_cyp.append(result.pred_boxes)
                else :
                    pred_masks_nuc.append(result.pred_masks)
                    pred_boxes_nuc.append(result.pred_boxes)

            idx_cyps = np.zeros(len(pred_masks_cyp))
            last_filename_idx = 0
            for i in range(len(pred_masks_nuc)) :
                box_nuc = np.array(pred_boxes_nuc[i])[0]
                box_nuc_centerCoor = [int((box_nuc[0] + box_nuc[2]) / 2), int((box_nuc[1] + box_nuc[3]) / 2)] # (x, y)
                minDist = 100000
                indexCyp = -1
                for j in range(len(pred_masks_cyp)) :
                    box_cyp = np.array(pred_boxes_cyp[j])[0]
                    box_cyp_centerCoor = [int((box_cyp[0] + box_cyp[2]) / 2), int((box_cyp[1] + box_cyp[3]) / 2)] # (x, y)
                    dist = math.sqrt(math.pow(box_nuc_centerCoor[0] - box_cyp_centerCoor[0], 2) + 
                                    math.pow(box_nuc_centerCoor[1] - box_cyp_centerCoor[1], 2))
                    if dist < minDist :
                        minDist = dist
                        indexCyp = j  
                kernel = np.ones((3,3), np.uint8)            
                # nuclei    
                pred_masks_nuc_dilate = cv2.dilate(pred_masks_nuc[i][0].cpu().numpy().astype(np.uint8), kernel, iterations = 3)
                pred_masks_nucCyp = 40 * pred_masks_nuc_dilate 
                overlapSize = 0
                if indexCyp != -1 :  
                    if minDist <= 400 :
                        idx_cyps[indexCyp] = 1
                        pred_masks_cyp_dilate = cv2.dilate(pred_masks_cyp[indexCyp][0].cpu().numpy().astype(np.uint8), kernel, iterations = 0) 
                    else :
                        kernel = np.ones((5,5), np.uint8)    
                        pred_masks_cyp_dilate = cv2.dilate(pred_masks_nuc_dilate, kernel, iterations = 10)     
                        print(fileName + "_" + str(minDist))        

                    pred_masks_nucCyp += 20 * pred_masks_cyp_dilate
                    
                    # calc overlapped box
                    box_minDistCyp = np.array(pred_boxes_cyp[indexCyp])[0]
                    overlapSize = getOverlapAreaSize(box_nuc, box_minDistCyp) 
                    
                    index = pred_masks_nucCyp[:,:] > 40
                    index_ = np.argwhere(index)
                    pred_masks_nucCyp[index_[:, 0], index_[:, 1]] = 40                    
                    pred_masks_nucCyp.astype(np.uint8)

                    # if overlapSize > 0:
                    #     pred_masks_nucCyp +=  20 * pred_masks_cyp_dilate
                    #     index = pred_masks_nucCyp[:,:] > 40
                    #     index_ = np.argwhere(index)
                    #     pred_masks_nucCyp[index_[:, 0], index_[:, 1]] = 40

                output_path = args.output                   
                fileName = file.split('/')[-1].split('.')[0] + "_" + str(i) + ".png"
                print(fileName + " box overlapped size : " + str(overlapSize))
                out_filename = output_path + '/' + fileName
                cv2.imwrite(out_filename, pred_masks_nucCyp)     
                last_filename_idx = i + 1

            for i in range(len(idx_cyps)) :
                if idx_cyps[i] == 0 :
                    box_cyp = np.array(pred_boxes_cyp[i])[0]
                    box_cyp_centerCoor = [int((box_cyp[0] + box_cyp[2]) / 2), int((box_cyp[1] + box_cyp[3]) / 2)] # (x, y)
                    minDist = 100000
                    indexNuc = -1
                    for j in range(len(pred_masks_nuc)) :
                        box_nuc = np.array(pred_boxes_nuc[j])[0]
                        box_nuc_centerCoor = [int((box_nuc[0] + box_nuc[2]) / 2), int((box_nuc[1] + box_nuc[3]) / 2)] # (x, y)
                        dist = math.sqrt(math.pow(box_nuc_centerCoor[0] - box_cyp_centerCoor[0], 2) + 
                                        math.pow(box_nuc_centerCoor[1] - box_cyp_centerCoor[1], 2))
                        if dist < minDist :
                            minDist = dist
                            indexNuc = j  
                    kernel = np.ones((3,3), np.uint8)            
                    # nuclei    
                    pred_masks_cyp_dilate = cv2.dilate(pred_masks_cyp[i][0].cpu().numpy().astype(np.uint8), kernel, iterations = 0)  
                    pred_masks_nucCyp = 20 * pred_masks_cyp_dilate       
                    if indexNuc != -1 :              
                        pred_masks_nuc_dilate = cv2.dilate(pred_masks_nuc[indexNuc][0].cpu().numpy().astype(np.uint8), kernel, iterations = 3)
                        pred_masks_nucCyp += 40 * pred_masks_nuc_dilate

                        index = pred_masks_nucCyp[:,:] > 40
                        index_ = np.argwhere(index)
                        pred_masks_nucCyp[index_[:, 0], index_[:, 1]] = 40                    
                        pred_masks_nucCyp.astype(np.uint8)

                    output_path = args.output                   
                    fileName = file.split('/')[-1].split('.')[0] + "_" + str(last_filename_idx) + ".png"
                    print(fileName + " box overlapped size : " + str(overlapSize))
                    out_filename = output_path + '/' + fileName
                    cv2.imwrite(out_filename, pred_masks_nucCyp)    
                    last_filename_idx += 1                

            logger.info(
                "{}: {} in {:.2f}s".format(
                    file,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )
            if args.output:                
                fileName = file.split('/')[-1].split('.')[0] + "_result1.jpg"
                out_filename = './result_all/' + fileName
                visualized_output.save(out_filename)

            