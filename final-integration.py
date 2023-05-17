import cv2
import math
import numpy as np
import dlib
import imutils
from imutils import face_utils
from matplotlib import pyplot as plt
import vlc
import webbrowser
import joblib

#loading pretrained model
svm_model = joblib.load('model3.pkl')

def yawn(mouth):
    return ((Euclidean_Distance(mouth[2], mouth[10])+Euclidean_Distance(mouth[4], mouth[8]))/(2*Euclidean_Distance(mouth[0], mouth[6])))

# Calculation of Euclidian Distance using mathematic concept
def Euclidean_Distance(a, b):
    return (math.sqrt(math.pow(a[0]-b[0], 2)+math.pow(a[1]-b[1], 2)))

# The function to detect eyes and determine if they are closed or open
def writeEyes(a, b, img, frame_count):
    margin = 15  # Adjusting this value can help to capture more or less surrounding area

    y1 = max(a[1][1] - margin, 0)
    y2 = min(a[4][1] + margin, img.shape[0])
    x1 = max(a[0][0] - margin, 0)
    x2 = min(a[3][0] + margin, img.shape[1])
    
    # Ensure it's a square image
    dy, dx = y2 - y1, x2 - x1
    delta = abs(dy - dx)
    
    if dy > dx:
        x1 -= delta // 2
        x2 += delta - delta // 2
    else:
        y1 -= delta // 2
        y2 += delta - delta // 2
    
    # Clip for safety
    y1, y2, x1, x2 = max(y1, 0), min(y2, img.shape[0]), max(x1, 0), min(x2, img.shape[1])
    
    left_eye_image = cv2.resize(img[y1:y2, x1:x2], (60, 60))
    cv2.imwrite('left-eye.jpg', left_eye_image)

    y1 = max(b[1][1] - margin, 0)
    y2 = min(b[4][1] + margin, img.shape[0])
    x1 = max(b[0][0] - margin, 0)
    x2 = min(b[3][0] + margin, img.shape[1])
    
    # Ensure it's a square image
    dy, dx = y2 - y1, x2 - x1
    delta = abs(dy - dx)
    
    if dy > dx:
        x1 -= delta // 2
        x2 += delta - delta // 2
    else:
        y1 -= delta // 2
        y2 += delta - delta // 2
    
    # Clip for safety
    y1, y2, x1, x2 = max(y1, 0), min(y2, img.shape[0]), max(x1, 0), min(x2, img.shape[1])
    
    right_eye_image = cv2.resize(img[y1:y2, x1:x2], (60, 60))
    cv2.imwrite('right-eye.jpg', right_eye_image)

    left_eye_drowsy = is_drowsy('left-eye.jpg', svm_model)
    right_eye_drowsy = is_drowsy('right-eye.jpg', svm_model)
    print(left_eye_drowsy,right_eye_drowsy)
    
    global drowsy_frame_count

    if left_eye_drowsy and right_eye_drowsy:
        drowsy_frame_count += 1
        if drowsy_frame_count > 10:
            cv2.putText(gray, "Drowsiness Detected", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
            print("Drowsiness detected!")
            alert.play()
    else:
        drowsy_frame_count = 0
        alert.stop()

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read the image in grayscale
    img = img.flatten()  # Flatten the image
    return img

def is_drowsy(image_path, svm_model):
    img = preprocess_image(image_path)
    prediction = svm_model.predict([img])
    return prediction[0] == 0

alert = vlc.MediaPlayer('focus.mp4')
alert1 = vlc.MediaPlayer('take_a_break.mp4')

yawn_counter = 0
drowsy_frame_count = 0

capture = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
print("predictor",predictor)
(leStart, leEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(reStart, reEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
frame_count = 0
yawn_detected = False

while(True):
    ret, frame = capture.read()
    size = frame.shape
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = frame
    rects = detector(gray, 0)
    if(len(rects)):
        shape = face_utils.shape_to_np(predictor(gray, rects[0]))
        leftEye = shape[leStart:leEnd]
        rightEye = shape[reStart:reEnd]
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)

        if(yawn(shape[mStart:mEnd])>0.8):
            cv2.putText(gray, "Yawn Detected", (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,127),2)
            if not yawn_detected:  # Only increment yawn_counter if a yawn was not already detected in the previous frame
                yawn_counter += 1
            yawn_detected = True  # Set yawn_detected to True since a yawn has been detected
        else:
            yawn_detected = False

        if yawn_counter >= 3:
            yawn_counter = 0  # Reset the yawn_counter
            alert1.play()
            webbrowser.open("https://www.google.com/maps/search/hotels+or+motels+near+me")

        writeEyes(leftEye, rightEye, frame, frame_count)
        frame_count += 1
    cv2.imshow('Driver', gray)
    if(cv2.waitKey(1)==27):
        break
        
capture.release()
cv2.destroyAllWindows()
