###########
# Kaan Acar
#
# 040150039
#
###########

import cv2
import numpy as np



######## Kalman Filter
class KalmanFilter:

    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def Estimate(self, coordX, coordY):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted



####### Name of the video !!!

video_name = "video.mp4"

# opening the video capture
cap = cv2.VideoCapture(video_name)

#first frame
_,first_frame = cap.read()
first_frame_gray = cv2.cvtColor(first_frame,cv2.COLOR_BGR2GRAY)

frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


buf = np.empty((frameCount, frameHeight, frameWidth), np.dtype('uint8'))
frames = np.empty((frameCount, frameHeight, frameWidth), np.dtype('uint8'))
three_frame_dif = np.empty((frameCount, frameHeight, frameWidth), np.dtype('uint8'))
final_video = np.empty((frameCount, frameHeight, frameWidth), np.dtype('uint8'))
merged = np.empty((frameCount, frameHeight, frameWidth), np.dtype('uint8'))




########################## Background subtraction and reading the frames
fc = 0
while True:

    _ , frame = cap.read()

    if(_ == False):
        break

    frame_gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    frames[fc] = frame_gray


    #Background subtraction
    difference = cv2.absdiff(first_frame_gray,frame_gray)
    ret , difference = cv2.threshold(difference,10,255,cv2.THRESH_BINARY)
    #we add the binary image to our empty matrix
    buf[fc] = difference
    fc += 1



############################### Three frame difference method
fc = 0
while (fc < frameCount):


    if((fc == 0) or (fc == (frameCount - 1))):
        three_frame_dif[fc] = frames[fc]
    else:

        D_1 = cv2.absdiff(frames[fc],frames[fc - 1])
        ret , D_1 = cv2.threshold(D_1,2,255,cv2.THRESH_BINARY)

        D_2 = cv2.absdiff(frames[fc + 1],frames[fc])
        ret2 , D_2 = cv2.threshold(D_2,2,255,cv2.THRESH_BINARY)

        result = np.bitwise_and(D_1,D_2)

        three_frame_dif[fc] = result

    fc += 1




fc = 0
################################### kernels for morphological operations
kernel = np.ones((15,15),np.uint8)
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
kernel3 = np.ones((25,25),np.uint8)


################################ opening video capture again !!!!
cap = cv2.VideoCapture(video_name)

##################### kalman filter object
kfObj = KalmanFilter()
predictedCoords = np.zeros((2, 1), np.float32)


#################### merging background subtraction and three frame dif method / morphology transformations

for i in range(0,frameCount):

    _,color = cap.read()

    ### Bitwise AND operation
    final_video[i] = cv2.bitwise_and(buf[i],three_frame_dif[i])
    merged[i] = final_video[i]

    ####  Morphological transformations
    final_video[i] = cv2.morphologyEx(final_video[i], cv2.MORPH_OPEN, kernel2)
    final_video[i] = cv2.dilate(final_video[i],kernel,iterations = 1)
    final_video[i] = cv2.morphologyEx(final_video[i], cv2.MORPH_CLOSE, kernel3)

    # finding contours
    contours,h = cv2.findContours(final_video[i],cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)

    for pic , contour in enumerate(contours):
        area = cv2.contourArea(contour)
        # finding blobs on the current frame
        if(area > 2500):
            x,y,w,h = cv2.boundingRect(contour)
            cv2.rectangle(color,(x,y),(x+w , y+h),(0,0,255 ),2)

            ## estimating possible position
            predictedCoords = kfObj.Estimate(x,y)

            x0 = int(predictedCoords[0])
            y0 = int(predictedCoords[1])

            # Drawing a rectangle as the predicted object position
            cv2.rectangle(color, (x0 , y0 ), (x0 + w, y0 + h), (255,0,0 ), 2)


            cv2.putText(color,"Current Position" ,(x + w, y + h), 0, 0.5, (0, 0, 255), 2)
            cv2.putText(color, "Predicted Position", (x0 + w, y0), 0, 0.5, (255, 0, 0), 2)


    cv2.imshow("window",color)
    key = cv2.waitKey(33)
    if(key == 27):
        break


##### comparison of different operations
#cv2.imwrite("background.png",buf[145])
#cv2.imwrite("three-frame.png",three_frame_dif[145])
#cv2.imwrite("merged.png",merged[145])
#cv2.imwrite("final.png",final_video[145])
#cv2.waitKey()

cap.release()
cv2.destroyAllWindows()
