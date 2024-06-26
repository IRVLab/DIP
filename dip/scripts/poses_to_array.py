# -*- coding: utf-8 -*-
"""Based on mediapipe_pose.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZDBHBemR37W_Ufng5mraYEU8K-sBJRkS

From usage example of MediaPipe Pose Solution API in Python (see also http://solutions.mediapipe.dev/pose).
"""
import cv2
import math
import numpy as np
import shutil
import os
from os import path


import mediapipe as mp

mp_pose = mp.solutions.pose


def get_image_path( ann_file, im_dir):
    ann_name = os.path.basename(ann_file)
    ann_name = os.path.splitext(ann_name)[0]
    image = os.path.join(im_dir, ann_name + '.jpg')
    return image

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480
def resize_and_show(image):
  h, w, z = np.shape(image)
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  cv2.imshow(img)
  cv2.waitKey(1000)

''' Get array of  2D landmark location in image pixel space
[RIGHT_SHOULDER, RIGHT_WRIST, RIGHT_HIP, LEFT_HIP, hip center] coordinates
'''
def get2d_poses(image):
  """All MediaPipe Solutions Python API examples are under mp.solutions.

  For the MediaPipe Pose solution, we can access this module as `mp_pose = mp.solutions.pose`.

  You may change the parameters, such as `static_image_mode` and `min_detection_confidence`, during the initialization. Run `help(mp_pose.Pose)` to get more informations about the parameters.
  """


  # Run MediaPipe Pose and draw pose landmarks.
  with mp_pose.Pose(
      static_image_mode=False, min_detection_confidence=0.5, model_complexity=2) as pose:
    points2d = []
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    image_height, image_width, _ = image.shape
    if not results.pose_landmarks:
      print('no pose')
      
      return []
      
    #Hip center
    else:
      hip_center_x = abs(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image_width-results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * image_width)/2+min(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image_width,results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * image_width)
      hip_center_y = abs(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image_height-results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * image_height)/2 + +min(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image_height, results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * image_height)
      
    
      points2d.append((hip_center_x,hip_center_y))
      # Rest of keypoints
      bodylandmarks = [mp_pose.PoseLandmark.RIGHT_SHOULDER,mp_pose.PoseLandmark.RIGHT_ELBOW,mp_pose.PoseLandmark.RIGHT_WRIST]#,mp_pose.PoseLandmark.RIGHT_HIP,mp_pose.PoseLandmark.LEFT_HIP,mp_pose.PoseLandmark.LEFT_SHOULDER,mp_pose.PoseLandmark.LEFT_ELBOW,mp_pose.PoseLandmark.LEFT_WRIST]
      for bodylandmark in bodylandmarks:
          points2d.append((results.pose_landmarks.landmark[bodylandmark].x * image_width,results.pose_landmarks.landmark[bodylandmark].y * image_height))
    return np.array(points2d)
 

if __name__ == '__main__':
  points2d_vector = get2d_poses()
