# IBM_challenge_Cricket_Pose_estimation

One of the most apparent dimensions applicable to pose estimation is
tracking and measuring human activity and movement. Many architectures
like OpenPose, PoseNet, and DensePose are often practiced for action,
gesture, or gait recognition. We have used human pose estimation in the
field of cricket for the following:

1. Identifying different batting shots
2. Check if a bowler has legal action or not.
3. Cricket Umpire signal Identification.

The innovation we propose in our solution is a special feature of
commentary. After the model has classified the shot, it will display the
name of the shot as well as pronounce the name of the shot. For example,
if a batsman hits a straight drive, the output will be displayed as "straight
drive" and also a voice note would be played stating that "It is a straight
drive". Similarly, this would also be applied to other models which will
identify whether the action of the bowler is legal, and the signal of the
umpire. This innovation would be a boon for visually impaired people who
would like to know which shot is being played or what is the signal of the
umpire.

## Outputs 



https://github.com/Devanshu-singh-VR/IBM_challenge_Cricket_Pose_estimation/assets/75822824/3757311c-5798-44ed-9976-2a54b4bce8b5



https://github.com/user-attachments/assets/8bf84b8d-0d8a-4d13-b02d-9e7a1e260e22



https://github.com/Devanshu-singh-VR/IBM_challenge_Cricket_Pose_estimation/assets/75822824/a0f1f0c9-157c-4c52-a59b-aa1d0ffa8fed



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

## The proposed solution required development of the following modules :

1. Data Extraction - We recorded videos of cricket shots and umpiring actions for
the dataset. Captured videos of different poses of different persons and store the
data for further processing. The following modules were implemented for the data
extraction.

- capture_data.py - This module is used to record the live video and store it
in a specific folder. We did this with the help of OpenCV Library.

- extract frame.py - This module breaks down the captured video into 32-
frame videos. Each video will be stored in a specific folder with a CSV file
that will contain the path of every video.

- data.csv - This CSV file contains a video path along with the class label
corresponding to each video. We labeled every video manually for
creating our dataset.

2. Data Preprocessing - We used Pytorch for data processing. The video should
be structured in a proper format for model training. The following modules were
implemented for the data preprocessing.

- data_lit.py - This module contains a video_get() function which will
convert the video data into a Pytorch tensor of size (no channels, frame,
width, height), the default size is (3, 32, 200, 200), and it also extracts
human body landmarks with the help of MediaPipe library which will be
further converted into Pytorch tensor of size (frame, no of landmarks). This
module will return the video tensor, landmark tensor, and labels
corresponding to every video.

3. Model - We implemented a Deep Learning hybrid model consisting of the
combination of 3D CNN and 1D CNN models.

- OneD_CNN.py - This is a 1D CNN model which will take human body
landmarks as an input of size (frames, total landmarks). This model
consists of one 1D CNN layer along with the maxpool layer, and three
linear layers. The output of the model will be size 120. The output of this
model will be used further in the 3D CNN module.

- ThreeD_CNN.py - This is a 3D CNN model which will take video as an
input of size (no channels, frames, width, height). This model consists of
five 3D CNN and maxpool layers along with the Batch Normalization and
three linear layers. The output of this model will be concatenated with the
output of the 1D CNN model, the size of the concatenated layer will be 240. 
The concatenated output will be passed through the final softmax
layer of size 9 (number of classes) for classification.

4. Training - We combined all the modules for training the model. We used Pytorch
for implementing and training the model. The following module we built for the
training.

- Train.py - This consists of a DataLoader function which will be used for the
data processing with the help data_lit.py module. The data will be
processed further into the CNN model and a Categorical Cross Entropy
loss function will be used to optimize the Adam optimization function. We
trained the model for 51 epochs. The model checkpoint will be saved
during the training. Batch size of 1 is used for the training data.

5. Testing - We used a video for testing the model both for cricket shots and
umpiring actions. The following module we implemented for the testing.

- test.py - This consists of a trained CNN model which will be used for the
testing. For testing input, we are using a video consisting of cricket shot
frames and umpiring actions frames. The module will highlight the action
which will be taken by the person in the video according to the predicted
class label.

![techstackIBM](https://user-images.githubusercontent.com/75822824/201369739-ebb270a2-1fdc-4ec6-b134-799ed031c08f.png)


