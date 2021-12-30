# GUIDING TO USE 3D HUMAN POSE ESTIMATION



Subject: **ABNORMAL DETECTION**

Professor: **Jeong, Hieyong**

Student Name: **Phan Kim Ngan**

ID: **217161**

Paper: **3D human pose estimation in video with temporal convolutions and semi-supervised training**

## Setup Source Code

+ Download source code and pre-trained models


```bash
git clone https://github.com/facebookresearch/VideoPose3D.git

cd VideoPose3D

mkdir checkpoint
cd checkpoint
wget https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_cpn.bin
wget https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_humaneva15_detectron.bin
wget https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin
cd ..
```

+ Install packages in Ubuntu

```bash
# Install ffmpeg for converting video format
sudo apt install ffmpeg
# Install video codec to play video mp4
sudo apt install ubuntu-restricted-extras
```



## Install Environment 

+ Install Anconda3

  https://www.anaconda.com/products/individual

+ Setup pose3d environment and  corresponding packages

```bas
# pose3d enviromment
conda activate base
conda create -y -c anaconda -n pose3d python=3.8 ipykernel

# kernel for jupyter-lab
conda activate pose3d
python -m pip install --upgrade pip
python -m ipykernel install --user --name pose3d --display-name "pose3d"

# pytorch
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# other packages
pip install matplotlib pillow scipy scikit-image scikit-learn pandas
pip install opencv-python opencv-contrib-python

# dectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```



## Structure of Project

```
VideoPose3D
├── common
│   └── ...
├── checkpoints
│   ├── pretrained_h36m_cpn.bin
│   ├── pretrained_humaneva15_detectron.bin
│   └── pretrained_h36m_detectron_coco.bin
├── data
│   ├── prepare_data_2d_custom.py
│   ├── data_2d_custom_myvideos.npz  <-- create custom dataset myvideos from detron2 keypoints (test.mp4.npz)
│   └── ...
├── images
│   └── ...
├── inference
│   ├── infer_video_d2.py
│   └── ...
├── run.py
├── inputs
│   ├── test.mp4      <-- input
│   └── ...
├── outputs
│   ├── test.mp4.npz  <-- keypoints infered from detron2
│   └── ...
└── ...
```



## Demo on custom video

+ Make **inputs** and **outputs** directories at **VideoPose3D**

+ Put demo video into inputs such as **test.mp4**
+ Activate **pose3d** environment
+ Infer keypoints from all videos in inputs by command

```bash
# inputs : VideoPose3D/inputs/test.mp4
# outputs: VideoPose3D/outputs/test.mp4.npz (keypoints of all frames)

# At VideoPose3D directory
cd inference
python infer_video_d2.py 									\
	--cfg COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml 	\
	--output-dir ../outputs 								\
	--image-ext mp4 										\
	../inputs
```

+ Create a custom dataset

```bash
# inputs : VideoPose3D/outputs/test.mp4.npz (keypoints of all frames)
# outputs: VideoPose3D/data/data_2d_custom_myvideos.npz (custom dataset myvideos from detron2 keypoints - test.mp4.npz)

# At VideoPose3D directory
cd data
python prepare_data_2d_custom.py -i ../outputs/ -o myvideos
```

+ Render a custom video and exporting coordinates

```bash
python run.py 										\
	-d custom 										\ # custom dataset
	-k myvideos 									\ # custom dataset containing keypoints of all preprocessing videos
	-arc 3,3,3,3,3									\ 
	-c checkpoint 									\ # checkpoint directory containing pretrained weights
	--evaluate pretrained_h36m_detectron_coco.bin 	\ # pretrained model using coco
	--render 										\
	--viz-subject test.mp4 							\  # filename in inputs directory
	--viz-action custom 							\  # custom dataset
	--viz-camera 0 									\
	--viz-video inputs/test.mp4 					\  # input video
	--viz-output outputs/output.mp4 				\  # output rendering video with 2D + 3D keypoints
	--viz-size 6
```

+ Final Result

![output](outputs/output.png)

![output](images/output.gif)

## References

```
@inproceedings{pavllo:videopose3d:2019,
  title={3D human pose estimation in video with temporal convolutions and semi-supervised training},
  author={Pavllo, Dario and Feichtenhofer, Christoph and Grangier, David and Auli, Michael},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```
