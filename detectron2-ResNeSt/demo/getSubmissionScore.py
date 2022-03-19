
#!/bin/python

import os
import glob
import cv2
import numpy as np
import argparse
import platform
import tqdm
import numba
from numba import jit

def coords(mask,res,type_):
    res_ = mask.shape[:2]
    if res[0] != res_[0] or res[1] != res_[1]:
        mask = cv2.resize(mask.astype(np.uint8),(res[1],res[0]),interpolation=cv2.INTER_NEAREST).astype(np.bool)
    x,y = np.where(mask)
    return ';'.join([','.join([str(i),str(j)]) for i,j in zip(x,y)])

parser = argparse.ArgumentParser(description="add the srouce directory path")
parser.add_argument('-s','--source',help="provide path of source directory")
parser.add_argument('-gt','--groundtruth',help="GT Path")
parser.add_argument('-nu','--nucleus',help="the label assigned to nucleus in the instance mask",type=int,default=40)
parser.add_argument('-cy','--cyto',help="the label assigned to cytoplasm in the instance mask",type=int,default=20)

args = parser.parse_args()

source = args.source
gt_path = args.groundtruth

global res
res = (1080,1440)

delim = '\\' if platform.system() == 'Windows' else '/'

if source[-1] != delim:
    source += delim

imgs = list(set([i.split(delim)[-1].split('_')[0] for i in glob.glob(source+'/*')]))
imgs = sorted(imgs)

imgs_gt = list(set([i.split(delim)[-1].split('_')[0] for i in glob.glob(gt_path+'/*')]))
imgs_gt = sorted(imgs_gt)
ins_gt = {}
for img in imgs_gt:
    ins_gt[img] = np.float32([cv2.resize(cv2.imread(ins,0).astype(np.float32),res[::-1],interpolation=cv2.INTER_NEAREST) for ins in glob.glob(gt_path+'/'+img+'_*')])

def iou(x,y):
    insec = np.logical_and(x,y)
    uni = np.logical_or(x,y)
    return np.sum(insec)/(np.sum(uni))

def getScore(gt_,ins_pred):
    pred = np.zeros(res,dtype=np.bool)
    result = 0.0
    for i in range(gt_.shape[0]):
        mask_gt = gt_[i] > 0
        iou_ = 0.0
        for ins_p in ins_pred:
            for m,n in ins_p[0]:
                try:
                    pred[int(m),int(n)] = True
                except:
                    pass
            for m,n in ins_p[1]:
                try:
                    pred[int(m),int(n)] = True
                except:
                    pass

            tm_iou = iou(mask_gt,pred)
            pred[:,:] = False
            if tm_iou > iou_:
                iou_ = tm_iou
        result += iou_
    return result

def calcScore(submission):
    score = 0.0
    cnt = 0
    #print(ins_gt.keys())
    i = 0
    for img in tqdm.tqdm(imgs_gt):
        try:
            curscore = getScore(ins_gt[img],submission[img])
            print(str(i) + " : " + str(curscore / ins_gt[img].shape[0]))
            score += curscore
        except:
            pass # img absent in prediction
        cnt += ins_gt[img].shape[0]
        i += 1
    return score  / cnt

def parse(submission):
    with open(submission,'r') as fp:
        data = fp.read().split('\n')
    print(len(data))
    for d in tqdm.tqdm(data):
        ins = d.split('\t')
        yield ins[0],[[[[l for l in k.split(',')] for k in j.split(';')] for j in i.split(' ')] for i in ins[1:]]
    
def evaluate(submission):
    sub = {}
    for im,ins in parse(submission):
        if im in imgs_gt:
            sub[im] = ins
    return calcScore(sub)

def genSubmission() :    
    submission = []
    for img in tqdm.tqdm(imgs):
        insts = glob.glob(source+img+'_*')
        res_nc = [img.split(delim)[-1]]
        for ins in insts:
            ins_mask = cv2.imread(ins,0)
            ins_nu = coords(ins_mask == args.nucleus,res,'n')
            ins_cy = coords(ins_mask == args.cyto,res,'c')
            res_nc.append(ins_nu+' '+ins_cy)
        submission.append('\t'.join(res_nc))

    with open('./submission.txt','w') as fp:
        print(len(submission))
        fp.write('\n'.join(submission))
        print("saved: ./submission.txt")

if __name__ == '__main__':
    genSubmission()
    sub = './submission.txt'
    #score = evaluate(sub)
    #print(score * 100)