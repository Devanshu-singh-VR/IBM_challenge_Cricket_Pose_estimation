# IBM_challenge_Cricket_Pose_estimation

One of the most apparent dimensions applicable to pose estimation is
tracking and measuring human activity and movement. Many architectures
like OpenPose, PoseNet, and DensePose are often practiced for action,
gesture, or gait recognition. We have used human pose estimation in the
field of cricket for the following:

1. Identifying different batting shots
2. Check if a bowler has legal action or not.
3. Cricket Umpire signal Identification.

## Proposed Solution

We created a hybrid model that contains a 3D Convolutional Neural
network and a 1D Convolutional Neural network, 3D CNN consists of five
3D convolutional layers with kernel size 3 and with padding, and five
Maxpooling layers without padding. It will take input as a video frame to
process, and the video will contain 32 frames, the size of the input video
frame will be (3, 32, 200, 200). During the process, the size of each next
output layer is half of the previous output layer, further, the layer will be
flattened and then the flattened layer will be passed to the next three feedforward layers.

The 1D CNN consists of one 1D Convolutional layer with kernel size 3 and
without padding followed by one Maxpool layer without padding. It will take
input as a 2D list of human body landmarks extracted from each frame of
the video input. The size of the input will be (32, 66) where 66 are the
separate x y pair of 33 landmarks which will be extracted with the help of
the MediaPipe library. After passing the input through the 1D CNN layer,
further, the layer will be flattened and the flattened layer will be passed to
the next three feed-forward layers. The last layer of both networks will be
of size 120, both layers will be concatenated and passed to the next feedforward layer for classification.

The final softmax layer is of the size of the number of classes, we are
using 9 classes for both Batting and Umpiring. We are using Adam
optimization which will make the algorithm converge towards the minima
faster, and the Categorical cross-entropy loss function for multiclass
classification.

![3D CNN model architecture](https://user-images.githubusercontent.com/75822824/201369082-4104ffb9-9627-4e3b-8e64-82281f6fe4e7.png)



