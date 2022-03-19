1. In ./mmdetection/ folder, Please refer to [get_started.md](docs/get_started.md) for installation of mmdetection library
2. In ./mmdetection/detectron2-ResNeSt/ folder, and Please refer to "https://detectron2.readthedocs.io/en/latest/tutorials/install.html#" for installation of detectron2 library
3. move input images into ./mmdetection/dataset/
4. Download model weights from "https://drive.google.com/file/d/12XltdbEhtMF3QU0saYvoiszEuWYVezSJ/view?usp=sharing" and unzip in ./mmdetection/weights/ folder 
5. In ./mmdetection/ folder, run getResult.sh
-> ./mmdetection/result/ : results of each network
-> ./mmdetection/merged/ : ensenbled results
6. In ./mmdetection/postprocessing/ folder, run "toySeg.m" file for postprocessing by using "Matlab"
7. In ./mmdetection/ folder, run ./demo/getFinalResult.py
-> ./mmdetection/result_final/ : results of final images
