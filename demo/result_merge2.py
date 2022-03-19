import sys
import os
import numpy as np
from os import walk
import cv2
 

input_dir = './result'
out_dir = './merged'
 
print("Working...")
    
# get all the pictures in directory
images = []
ext = (".jpeg", ".jpg", ".png", ".bmp")

global res
res = (1080,1440)
 
for (dirpath, dirnames, filenames) in walk(input_dir):
    for filename in filenames:
        if filename.endswith(ext):
            images.append(os.path.join(dirpath, filename))
 
idx_idx = []
idx_filename = []
for image in images:
    fileName = image.split('/')[-1].split('.')[0]
    imgName = fileName.split('_')[0]

    if imgName in idx_idx: 
        idx_tmp = idx_idx.index(imgName)
        idx_filename[idx_tmp].append(image)
    else : 
        idx_idx.append(imgName)
        idx_filename.append([])
        idx_tmp = idx_idx.index(imgName)
        idx_filename[idx_tmp].append(image)

for i in range (len(idx_filename)) :
    idx_resultfile = 0
    idx_remove = []
    for j in range (len(idx_filename[i])) :
        if j not in idx_remove :
            print(idx_filename[i][j])
            img_tgt = cv2.imread(idx_filename[i][j], cv2.IMREAD_UNCHANGED) 
            img_tgt = cv2.resize(img_tgt,res[::-1],interpolation=cv2.INTER_NEAREST)
            h, w = img_tgt.shape[:2]
            idx_overlap = []
            for k in range (len(idx_filename[i])) :
                if j != k :   
                    img_ref = cv2.imread(idx_filename[i][k], cv2.IMREAD_UNCHANGED)
                    img_ref = cv2.resize(img_ref,res[::-1],interpolation=cv2.INTER_NEAREST)
                    idx1 =  img_tgt[:,:] == 20
                    idx2 =  img_ref[:,:] == 20
                    a = np.multiply(idx1, idx2)
                    b = np.add(idx1, idx2)
                    ratio = np.sum(np.array(a)) / np.sum(np.array(b))
                    
                    idx1 =  img_tgt[:,:] == 40
                    idx2 =  img_ref[:,:] == 40
                    a = np.multiply(idx1, idx2)
                    b = np.add(idx1, idx2)
                    ratio2 = np.sum(np.array(a)) / np.sum(np.array(b))
                    if ratio >= 0.9 and ratio2 >= 0.9 :
                        idx_overlap.append(k)
                        idx_remove.append(k)
            img_new = img_tgt
            for k2 in idx_overlap :
                print("merged - " + idx_filename[i][k2])
            #     img_ref = cv2.imread(idx_filename[i][k2], cv2.IMREAD_UNCHANGED)
            #     img_ref = cv2.resize(img_ref,res[::-1],interpolation=cv2.INTER_NEAREST)
            #     idx1_1 = np.multiply(img_tgt[:,:] == 0, img_ref[:,:] == 20)
            #     idx1_2 = np.multiply(img_tgt[:,:] == 20, img_ref[:,:] == 0)
            #     idx1_3 = np.multiply(img_tgt[:,:] == 20, img_ref[:,:] == 20)
            #     idx1 = np.add(idx1_1, np.add(idx1_2, idx1_3))
            #     idx1 = np.argwhere(idx1)        
            #     idx2_1 = np.multiply(img_tgt[:,:] == 40, img_ref[:,:] == 20)
            #     idx2_2 = np.multiply(img_tgt[:,:] == 20, img_ref[:,:] == 40)
            #     idx2_3 = np.multiply(img_tgt[:,:] == 40, img_ref[:,:] == 40)
            #     idx2 = np.add(idx2_1, np.add(idx2_2, idx2_3))
            #     idx2 = np.argwhere(idx2)        
            #     img_new[idx1[:, 0], idx1[:, 1]] = 20
            #     img_new[idx2[:, 0], idx2[:, 1]] = 40

            fileName = idx_filename[i][j].split('/')[-1].split('.')[0]
            imgName = fileName.split('_')[0]
            saveName = out_dir + '/' + imgName + '_' + str(idx_resultfile) + '.png'
            idx_resultfile += 1
            cv2.imwrite(saveName, img_new)

print("Completed!")