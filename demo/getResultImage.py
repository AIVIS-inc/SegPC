from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector, show_result_pyplot
import numpy as np
import os
from PIL import Image, ImageOps
import pycocotools.mask as maskUtils
from fvcore.common.file_io import PathManager
import mmcv
import cv2
import math


def allfiles(path):
    res = []
    
    for root, dirs, files in os.walk(path):
        for file in files:
            filepath = os.path.join(root, file)
            res.append(filepath)

    return res

def read_image(file_name, format=None):
    """
    Read an image into the given format.
    Will apply rotation and flipping if the image has such exif information.

    Args:
        file_name (str): image file path
        format (str): one of the supported image modes in PIL, or "BGR"

    Returns:
        image (np.ndarray): an HWC image in the given format.
    """
    with PathManager.open(file_name, "rb") as f:
        image = Image.open(f)

        # capture and ignore this bug: https://github.com/python-pillow/Pillow/issues/3973
        try:
            image = ImageOps.exif_transpose(image)
        except Exception:
            pass

        if format is not None:
            # PIL only supports RGB, so convert to RGB and flip channels over below
            conversion_format = format
            if format == "BGR":
                conversion_format = "RGB"
            image = image.convert(conversion_format)
        image = np.asarray(image)
        if format == "BGR":
            # flip channels if needed
            image = image[:, :, ::-1]
        # PIL squeezes out the channel dimension for "L", so make it HWC
        if format == "L":
            image = np.expand_dims(image, -1)
        return image

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('output', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.01, help='bbox score threshold')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)    
    for file in allfiles(args.img):
        result = inference_detector(model, file)
        # show the results
        #show_result_pyplot(model, args.img, result, score_thr=args.score_thr)
        img = mmcv.imread(file)
        img = img.copy()

        bbox_result, segm_result = result
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        segms = None
        if segm_result is not None:    
            segms = mmcv.concat_list(segm_result)
            inds = np.where(bboxes[:, -1] > args.score_thr)[0]
            # np.random.seed(42)
            # color_masks = [
            #     np.random.randint(0, 256, (1, 3), dtype=np.uint8)
            #     for _ in range(max(labels) + 1)
            # ]
            # for i in inds:
            #     i = int(i)
            #     color_mask = color_masks[labels[i]]
            #     mask = maskUtils.decode(segms[i]).astype(np.bool)
            #     img[mask] = img[mask] * 0.5 + color_mask * 0.5

            pred_masks_nuc = []
            pred_masks_cyp = []
            pred_boxes_nuc = []
            pred_boxes_cyp = []
            for i in inds:
                mask = segms[i].astype(np.bool)
                if labels[i] == 0:
                    pred_masks_cyp.append(mask)
                    pred_boxes_cyp.append(bboxes[i])
                else :
                    pred_masks_nuc.append(mask)
                    pred_boxes_nuc.append(bboxes[i])
            idx_cyps = np.zeros(len(pred_masks_cyp))
            last_filename_idx = 0
            for i in range(len(pred_masks_nuc)) :
                box_nuc_centerCoor = [int((pred_boxes_nuc[i][0] + pred_boxes_nuc[i][2]) / 2), int((pred_boxes_nuc[i][1] + pred_boxes_nuc[i][3]) / 2)] # (x, y)
                minDist = 100000
                indexCyp = -1
                for j in range(len(pred_masks_cyp)) :
                    box_cyp_centerCoor = [int((pred_boxes_cyp[j][0] + pred_boxes_cyp[j][2]) / 2), int((pred_boxes_cyp[j][1] + pred_boxes_cyp[j][3]) / 2)] # (x, y)
                    dist = math.sqrt(math.pow(box_nuc_centerCoor[0] - box_cyp_centerCoor[0], 2) + 
                                    math.pow(box_nuc_centerCoor[1] - box_cyp_centerCoor[1], 2))
                    if dist < minDist :
                        minDist = dist
                        indexCyp = j  
                kernel = np.ones((3,3), np.uint8)            
                # nuclei    
                pred_masks_nuc_ = np.array(pred_masks_nuc[i]).astype(np.uint8)
                pred_masks_nuc_dilate = cv2.dilate(pred_masks_nuc_, kernel, iterations = 3)
                pred_masks_nucCyp = 40 * pred_masks_nuc_dilate 
                overlapSize = 0
                if indexCyp != -1 :  
                    if minDist <= 400 :
                        idx_cyps[indexCyp] = 1
                        pred_masks_cyp_dilate = cv2.dilate(np.array(pred_masks_cyp[indexCyp]).astype(np.uint8), kernel, iterations = 0)
                    else :
                        kernel = np.ones((5,5), np.uint8)    
                        pred_masks_cyp_dilate = cv2.dilate(pred_masks_nuc_dilate, kernel, iterations = 10)     
                        print(fileName + "_" + str(pred_boxes_nuc[i][4]) + "_" + str(minDist))        
                        
                    pred_masks_nucCyp += 20 * pred_masks_cyp_dilate

                    # idx1 = pred_masks_nuc_[:,:] == 0
                    # idx2 = pred_masks_nuc_dilate[:,:] == 1
                    # idx3 = pred_masks_cyp_dilate[:,:] == 1
                    # index = np.multiply(idx1, idx2)
                    # index = np.multiply(index, idx3)
                    # index_ = np.argwhere(index)
                    # pred_masks_nucCyp[index_[:, 0], index_[:, 1]] = 20      

                    index = pred_masks_nucCyp[:,:] > 40
                    index_ = np.argwhere(index)
                    pred_masks_nucCyp[index_[:, 0], index_[:, 1]] = 40                    
                    pred_masks_nucCyp.astype(np.uint8)

                output_path = args.output                   
                fileName = file.split('/')[-1].split('.')[0] + "_" + str(i) + ".png"
                print(fileName + "_" + str(pred_boxes_nuc[i][4]))
                out_filename = output_path + '/' + fileName
                cv2.imwrite(out_filename, pred_masks_nucCyp)     
                last_filename_idx = i + 1

            for i in range(len(idx_cyps)) :
                if idx_cyps[i] == 0 :
                    box_cyp_centerCoor = [int((pred_boxes_cyp[i][0] + pred_boxes_cyp[i][2]) / 2), int((pred_boxes_cyp[i][1] + pred_boxes_cyp[i][3]) / 2)] # (x, y)
                    minDist = 100000
                    indexNuc = -1
                    for j in range(len(pred_masks_nuc)) :
                        box_nuc_centerCoor = [int((pred_boxes_nuc[j][0] + pred_boxes_nuc[j][2]) / 2), int((pred_boxes_nuc[j][1] + pred_boxes_nuc[j][3]) / 2)] # (x, y)
                        dist = math.sqrt(math.pow(box_nuc_centerCoor[0] - box_cyp_centerCoor[0], 2) + 
                                        math.pow(box_nuc_centerCoor[1] - box_cyp_centerCoor[1], 2))
                        if dist < minDist :
                            minDist = dist
                            indexNuc = j  
                    kernel = np.ones((3,3), np.uint8)            
                    # nuclei    

                    pred_masks_cyp_dilate = cv2.dilate(np.array(pred_masks_cyp[i]).astype(np.uint8), kernel, iterations = 0)  
                    pred_masks_nucCyp = 20 * pred_masks_cyp_dilate      
                    if indexNuc != -1 :               
                        pred_masks_nuc_dilate = cv2.dilate(np.array(pred_masks_nuc[indexNuc]).astype(np.uint8), kernel, iterations = 3)
                        pred_masks_nucCyp += 40 * pred_masks_nuc_dilate

                        index = pred_masks_nucCyp[:,:] > 40
                        index_ = np.argwhere(index)
                        pred_masks_nucCyp[index_[:, 0], index_[:, 1]] = 40                    
                        pred_masks_nucCyp.astype(np.uint8)

                    output_path = args.output                   
                    fileName = file.split('/')[-1].split('.')[0] + "_" + str(last_filename_idx) + ".png"
                    print(fileName + "_" + str(pred_boxes_cyp[i][4]))
                    out_filename = output_path + '/' + fileName
                    cv2.imwrite(out_filename, pred_masks_nucCyp)    
                    last_filename_idx += 1 
            
        
        # out_file = None
        # mmcv.imshow_det_bboxes(
        # img,
        # bboxes,
        # labels,
        # class_names=None,
        # score_thr=args.score_thr,
        # show=False,
        # wait_time=0,
        # out_file=out_file)

        # fileName = file.split('/')[-1].split('.')[0] + ".png"
        # output_path = args.output + '/' + fileName
        # cv2.imwrite(output_path, img)     


if __name__ == '__main__':
    main()