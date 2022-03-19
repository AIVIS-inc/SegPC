#!/bin/bash

mkdir ./result

mkdir ./result/g_4
mkdir ./result/f_3_2
mkdir ./result/e_3_1
mkdir ./result/d_3_0
mkdir ./result/c_2_1
mkdir ./result/b_2_0
mkdir ./result/a_1_0
mkdir ./merged

python ./demo/getResultImage.py ./dataset ./result/g_4 ./models/1_4_2x_2.py ./weights/1_4_2x_2.pth
python ./demo/getResultImage.py ./dataset ./result/f_3_2 ./models/1_3_2x_6.py ./weights/1_3_2x_6.pth
python ./demo/getResultImage.py ./dataset ./result/e_3_1 ./models/1_3_2x_5.py ./weights/1_3_2x_5.pth
python ./demo/getResultImage.py ./dataset ./result/d_3_0 ./models/1_3_2x_4.py ./weights/1_3_2x_4.pth
python ./demo/getResultImage.py ./dataset ./result/c_2_1 ./models/1_1_2x_2_5.py ./weights/1_1_2x_2_5.pth
python ./demo/getResultImage.py ./dataset ./result/b_2_0 ./models/1_1_2x_2_4.py ./weights/1_1_2x_2_4.pth
python ./detectron2-ResNeSt/demo/getResultImage.py --config-file ./models/1_2_2x_4.yaml --input ./dataset --output ./result/a_1_0 --opts MODEL.WEIGHTS ./weights/1_2_2x_4.pth

python ./demo/result_merge2.py