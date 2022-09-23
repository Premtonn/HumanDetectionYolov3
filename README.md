# HumanDetectionYolov3

## Intro

This is a project that is concerned with Human Detection using the YOLO algorithm. It is able to detect humans based on their body features, and it counts all humans that
are within the frame. The project is also setup with the flask framework, enabling browser streaming of the camera feed.

## Prerequisites

Before you use the code, you will need to install OpenCV ```pip install opencv-python```. Next step is instsalling numpy ```pip install numpy```, then you have to also install flask ```pip install Flask```. You will aslo need to download yolov3.weights and put it in the right directory. You can download it from this link: 
https://pjreddie.com/darknet/yolo/


## Side Note

The code will run better and faster on GPU than on CPU. In order to do that, you need to install cuda and the cuDNN version that is compatible with your GPU. After that you will need to download and install CMake and Visual Studio with a C++ environment. To finish this whole process watch the following video, which will explain everything thoroughly: https://www.youtube.com/watch?v=YsmhKar8oOc


