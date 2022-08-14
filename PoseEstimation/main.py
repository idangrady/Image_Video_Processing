import cv2
import mediapipe as mp
import numpy as np


mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose= mp_pose.Pose()

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()

    result = pose.process(img)
    mp_draw.draw_landmarks(img,result.pose_landmarks, mp_pose.POSE_CONNECTIONS)


    Op_IMg = np.zeros([600,400,3])
    Op_IMg.resize(img.shape)
    Op_IMg.fill(255)
    mp_draw.draw_landmarks(Op_IMg, result.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_draw.DrawingSpec((0,255,0), 2),
                           mp_draw.DrawingSpec((255,0,0), 2))

    cv2.imshow("Pose Estimation", img)
    cv2.imshow("Extracted", Op_IMg)
    cv2.waitKey(1)


print("Yes")