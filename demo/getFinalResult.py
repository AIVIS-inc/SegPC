import sys
import os
import numpy as np
from os import walk
import cv2
 
in_dir1 = './merged'
in_dir2 = './merged_postprocessing'
out_dir = './result_final'
 
print("Working...")
    
# get all the pictures in directory
images = []
tmp_out = []
ext = (".jpeg", ".jpg", ".png", ".bmp")

def allfiles(path):
    res = []
    
    for root, dirs, files in os.walk(path):
        for file in files:
            filepath = os.path.join(root, file)
            res.append(filepath)

    return res

global res
res = (1080,1440)
 
for (dirpath, dirnames, filenames) in walk(in_dir1):
    for filename in filenames:
        if filename.endswith(ext):
            images.append(os.path.join(dirpath, filename))
 
idx_idx = []
idx_filename = []
cnt_idx = []
cnt = -1
for image in images:
    fileName = image.split('/')[-1].split('.')[0]
    imgName = fileName.split('_')[0]

    if imgName in idx_idx: 
        idx_tmp = idx_idx.index(imgName)
        idx_filename[idx_tmp].append(image)
        cnt += 1
    else : 
        idx_idx.append(imgName)
        if(cnt != -1):
            cnt_idx.append(cnt)
        cnt = 0
        idx_filename.append([])
        idx_tmp = idx_idx.index(imgName)
        idx_filename[idx_tmp].append(image)
cnt_idx.append(cnt)

for (dirpath, dirnames, filenames) in walk(in_dir2):
    for filename in filenames:
        if filename.endswith(ext):
            tmp_out.append(os.path.join(dirpath, filename))


for i in range(len(idx_idx)):
    cnt = 1
    for tmp in tmp_out:
        tmpNum = tmp.split('/')[-1].split('.')[0].split('_')[0]

        if idx_idx[i] == tmpNum :
            img_seg = cv2.imread(tmp, cv2.IMREAD_UNCHANGED) 
            filename_new = out_dir + '/' + tmpNum + '_' + str(int(cnt_idx[i]) + cnt) + '.png'
            cv2.imwrite(filename_new, img_seg)
            cnt += 1

        
for file in allfiles(out_dir):
    if file.split('.')[-1] == 'png':
        print(file)
        
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        fileName = file.split('\\')[-1].split('.')[0]
        imgName = fileName.split('_')[0]
        
        img_newGT = np.zeros((nH, nW), np.uint8)

        index = np.multiply(img[:,:] > 10, img[:,:] <= 30)
        index_ = np.argwhere(index)        
        img_newGT[index_[:, 0], index_[:, 1]] = 20
        
        if int(imgName) > 1000 : 
            img_tmp = np.zeros((nH, nW), np.uint8)
            index = img > 30
            index_shift = np.zeros((nH, nW), np.bool)
            index_shift[:, 6:nW] = index[:, 0:nW-6]    
            index_ = np.argwhere(index_shift)      
            img_newGT[index_[:, 0], index_[:, 1]] = 40
            inv_index = np.logical_not(index_shift)
            index_blank = np.multiply(index, inv_index)
            index_ = np.argwhere(index_blank)      
            img_newGT[index_[:, 0], index_[:, 1]] = 20
        else :
            index = img > 30
            index_ = np.argwhere(index)        
            img_newGT[index_[:, 0], index_[:, 1]] = 40

        cv2.imwrite(file, img_newGT)
