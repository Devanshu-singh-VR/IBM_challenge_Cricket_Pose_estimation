import cv2 as cv
import pandas as pd

datasheet = {'videos': [], 'labels': []}
frames = 32
video = cv.VideoCapture('batting/video6.mp4')
count = 26
fm = 0

while True:
    ret, img = video.read()

    # Set the new file path
    if fm == 0:
        fourcc = cv.VideoWriter_fourcc('X', 'V', 'I', 'D')
        videoWriter = cv.VideoWriter('Demo_bat/video' + str(count) + '.mp4', fourcc, 30.0, (640, 480))

    # frame condition (32 frames)
    if fm <= frames:
        videoWriter.write(img)
        fm += 1
    else:
        fm = 0
        datasheet['videos'].append('video' + str(count) + '.mp4')
        datasheet['labels'].append(0)
        count += 1

    # stop when video is over
    if str(img) == "None":
       break

print(count)
dataFrame = pd.DataFrame(datasheet)
dataFrame.to_csv('data1.csv', index=False)