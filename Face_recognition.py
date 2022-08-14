import cv2
from random import randrange

#load some pre-trained data on face frontals from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choose an image to detect faces in
#img = cv2.imread('LG.png')

# to capture video from webcam
webcam = cv2.VideoCapture(0)
# iterate forever over frames
while True:
    # read the current file
    successful_frame_read, frame = webcam.read()
    # converting to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    #draw rectangles around the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(255),randrange(255),randrange(255)), 2)

    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1)

    # stop if q key is pressed
    if key==81 or key==113:
        break

#release the videocapture object
webcam.release()
