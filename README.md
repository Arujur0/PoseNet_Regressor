# PoseNet
A fundamental implementation of PoseNet, a type of pose regression model.

## Introduction
At its core, PoseNet is a convolutional neural network and uses convolutional layers to learn the features of the training images. These features are then used to learn specific properties present in the images, such as the camera position. CNNs tend to be very deep and hard to train, especially without large GPUs. Because of this, PoseNet relies on a pre-trained "general-purpose feature extractor".

## Architecture
The architecture of the PoseNet consists of a series of InceptionBlocks that lead into three loss headers. Each of these loss headers predicts an xyz position and a wpqr orientation. The position is predicted as a 3D coordinate and the orientation as a Quaternion in wpqr ordering.

## Dataset 
The neural network is trained on the KingsCollege Dataset. It is a large scale outdoor localisation dataset. It consists of video recordings and images of buildings around Cambridge University. The dataset can be downloaded from [here](https://www.repository.cam.ac.uk/bitstream/handle/1810/251342/KingsCollege.zip).

## Implementation

## Results


