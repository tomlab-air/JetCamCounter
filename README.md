![top_image_small2](https://github.com/user-attachments/assets/ef3a3e48-1136-4012-99f2-6cd36a69b199)

# JetCamCounter

JetCamCounter is a real-time object detection & counting system running on NVIDIA Jetson. Using YOLOv4-tiny, it detects and tracks objects such as people and cars from a camera or video file and automatically counts the number of crossings.

It is designed to replace traffic volume surveys with AI. It can also be applied to other use cases (entry/exit management, headcounting, security, store analysis, etc.).

![person2AI2](https://github.com/user-attachments/assets/d2ac3db8-9a73-4508-bd0b-80800ccd0235)

<br/>

## 📹 Demo

https://github.com/user-attachments/assets/a51f9eac-a9ab-4847-a90a-51b164dc3cf2

- Counts objects that cross the central vertical line and displays the count in the top left corner
- Supports both camera feed and video file
- Outputs results to a video file and a text file for later review

<br/>

## 🚀 Background and Purpose

Traffic volume surveys have traditionally been conducted manually, which presents the following challenges:

- Requires long hours of work
- Risk of human error
- Manual effort required to digitize data
- High cost associated with hiring personnel

JetCamCounter leverages the edge AI performance of Jetson and the lightweight nature of YOLOv4-tiny to build a system that **processes camera footage in real time** and automatically counts crossings. With the low power consumption and compact size of Jetson, **long-duration outdoor operation is possible**. Additionally, because there is no need for a person to stand by and count, labor costs are reduced.

<br/>

## 🔍 Features

- ✅ Jetson compatible, low power consumption, high efficiency
- ✅ Fast detection using YOLOv4-tiny
- ✅ Switch between camera or video file input
- ✅ Count crossings using tracking
- ✅ Supports arbitrary object classes (cars, people, etc.)

<br/>

## 🧠 Use Cases

- 🚗 **Traffic volume surveys** (counting vehicles at intersections)
- 🏢 **Entry/exit management** (headcounting in offices or facilities)
- 🛍 **Retail store analysis** (analyzing customer behavior)
- 🧍‍♂️ **Queue length estimation** (monitoring number of people waiting)
- 🔐 **Security applications** (detecting unauthorized intrusions)

<br/>

## ⚙️ Setup Instructions (for Jetson)

### 1. Initial Jetson Setup

Refer to resources such as the following to perform the initial Jetson setup:  
[Getting Started with AI on Jetson Nano](https://learn.nvidia.com/courses/course-detail?course_id=course-v1:DLI+S-RX-02+V2)

### 2. Install PyTorch for Jetson
Follow the instructions on the page below to install PyTorch for Jetson:  
[PyTorch for Jetson](https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048)

### 3. Install Python Dependencies
```bash
sudo apt install -y libopencv-dev python3-opencv
pip3 install numpy
```

### 4. Build Darknet (YOLOv4-tiny)
```bash
git clone https://github.com/AlexeyAB/darknet.git
cd darknet
# Edit the Makefile (enable GPU=1, CUDNN=1, OPENCV=1, etc.)
sed -i 's/GPU=0/GPU=1/' Makefile
sed -i 's/CUDNN=0/CUDNN=1/' Makefile
sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile
sed -i 's/OPENCV=0/OPENCV=1/' Makefile
sed -i 's/LIBSO=0/LIBSO=1/' Makefile
sed -i 's/^ARCH=.*/ARCH= -gencode arch=compute_53,code=[sm_53,compute_53]/' Makefile
# make
make -j$(nproc)
```

### 5. Download Weights and Config Files
```bash
# Download cfg files and pretrained weights
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg -P cfg/
wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/coco.data -P cfg/
wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
```

### 6. Copy main.py to the darknet folder
Copy `main.py` to the `darknet` folder.

<br/>

## ▶️ Usage

Run JetCamCounter with the following commands:

### Run with camera input
```bash
python3 main.py
```

### Run with video file input
```bash
python3 main.py input_video.mp4
```

Generated files during execution:  
- Output video: `result/<base_name>_<mode>_<timestamp>.mp4`  
- Count log: `result/<base_name>_<mode>_count_log_<timestamp>.txt`

<br/>

## 💡 Future Plans
- Automatically upload result files to a server
- Support for nighttime/infrared cameras
- Continuous improvement of detection accuracy
