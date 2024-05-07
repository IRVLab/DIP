# !/usr/bin/env python3  

from tkinter import Widget
import cv2
import os
import numpy as np
import poses_to_array
import glob
from matplotlib import pyplot as plt
import rospy
import roslib


def detect_object(left_img, right_img):

  size = left_img.shape
  mask = np.zeros(left_img.shape[:2], dtype=np.uint8)
  
  #Check mediapipe landmarks and reshape for both image
  left_points2d_array = poses_to_array.get2d_poses(left_img)
  right_points2d_array = poses_to_array.get2d_poses(right_img)

  # Set Triangle Information
  half_base = 100

  if len(left_points2d_array) != 0 and len(right_points2d_array) != 0:
    #Get mediapipe landmarks and reshape for left image
    shape2d,m = left_points2d_array.shape
    
    #Get Right image info
    shape2d,m = right_points2d_array.shape
        
    '''Find mean landmark location'''
    mean_2d_array = (right_points2d_array + left_points2d_array)/2

    #extend pointing in 3d
    '''3d line  extension elbow and wrist stuff'''
    
    # Left image info
    left_endpoint = point_extension(left_points2d_array[2],left_points2d_array[3])

    # right image info
    right_endpoint = point_extension(right_points2d_array[2],right_points2d_array[3])

    #Mean info
    mean_endpoint = point_extension(mean_2d_array[2],mean_2d_array[3])
    
    '''Find triangle from wrist'''
    #Mean endpoint gives height 
    # triangle base length: 1/2 is distance from endpint 10
    # 
    half_base = 100 
 

    triangle = np.array([[mean_endpoint[0],mean_endpoint[1]+half_base],[mean_endpoint[0],mean_endpoint[1]-half_base],[left_points2d_array[3][0]-5,left_points2d_array[3][1]+5]],dtype=np.int32)
    cv2.fillPoly(mask, pts=[triangle], color=(255, 255, 255))
    
    # keypoints
    sift_bbox = [-1,-1,-1,-1]
    sift = cv2.xfeatures2d.SIFT_create()
    kp_sift, descriptors_1 = sift.detectAndCompute(left_img,mask)
    pts = cv2.KeyPoint_convert(kp_sift)
    bbox_half = 10
    if pts != ():
      # # print(pts[0].size)
      '''Get bounding box
      float32            top_left_x
      float32            top_left_y
      float32            width
      float32            height'''
      top_left_x = int(pts[0][0] - bbox_half)

      if top_left_x < 0:
        top_left_x = 0
      
      top_left_y = int(pts[0][1] - bbox_half)
      if top_left_y < 0:
        top_left_y = 0
      width = bbox_half*2
      height = bbox_half*2

      # bottom_right_x = int(pts[0][0] + 50)
      # bottom_right_y = int(pts[0][1] + 50)

      sift_bbox = [top_left_x,top_left_y,width,height]
    sift_bbox = sift_bbox

    return [sift_bbox,left_points2d_array, triangle]
 
  elif len(left_points2d_array) != 0:
    #Get mediapipe landmarks and reshape for left image
    shape2d,m = left_points2d_array.shape
   
    # Left image info
    left_endpoint = point_extension(left_points2d_array[2],left_points2d_array[3])


    '''Find triangle from wrist'''
    #Mean endpoint gives height 
    # triangle base length: 1/2 is distance from endpint 10
    triangle = np.array([[left_endpoint[0],left_endpoint[1]+half_base],[left_endpoint[0],left_endpoint[1]-half_base],[left_points2d_array[3][0]-5,left_points2d_array[3][1]+5]],dtype=np.int32)

    cv2.fillPoly(mask, pts=[triangle], color=(255, 255, 255))
    #keypoints
    sift = cv2.xfeatures2d.SIFT_create()
    kp_sift, descriptors_1 = sift.detectAndCompute(left_img,mask)


    pts = cv2.KeyPoint_convert(kp_sift)
    
    if pts != ():
      '''Get bounding box
      float32            top_left_x
      float32            top_left_y
      float32            width
      float32            height'''
      top_left_x = int(pts[0][0] - 50)
      
      top_left_y = int(pts[0][1] - 50)
      width = 100
      height = 100

      bottom_right_x = int(pts[0][0] + 50)
      bottom_right_y = int(pts[0][1] + 50)

      bbox = [top_left_x,top_left_y,width,height]
 
      return [bbox, left_points2d_array, triangle] 
    else:
      return [[-1,-1,-1,-1],[], []]
  
  elif len(right_points2d_array) != 0:
    #Get mediapipe landmarks and reshape for right image
    #Get Right image info
    shape2d,m = right_points2d_array.shape   
   
    # right image info
    right_endpoint = point_extension(right_points2d_array[2],right_points2d_array[3])
    
    '''Find triangle from wrist'''
    #Mean endpoint gives height 
    # triangle base length: 1/2 is distance from endpint 10
    triangle = np.array([[right_endpoint[0],right_endpoint[1]+half_base],[right_endpoint[0],right_endpoint[1]-half_base],[right_points2d_array[3][0]-5,right_points2d_array[3][1]+5]],dtype=np.int32)

    
    cv2.fillPoly(mask, pts=[triangle], color=(255, 255, 255))

    #keypoints
    sift = cv2.xfeatures2d.SIFT_create()
    kp_sift, descriptors_1 = sift.detectAndCompute(left_img,mask)


    pts = cv2.KeyPoint_convert(kp_sift)
    print("points",pts)
    
    if pts != ():
      '''Get bounding box
      float32            top_left_x
      float32            top_left_y
      float32            width
      float32            height'''
      top_left_x = int(pts[0][0] - 50)
      
      top_left_y = int(pts[0][1] - 50)
      width = 100
      height = 100

      bottom_right_x = int(pts[0][0] + 50)
      bottom_right_y = int(pts[0][1] + 50)

      bbox = [top_left_x,top_left_y,width,height]

      return [bbox, left_points2d_array, triangle] 
  else:
    return [[-1,-1,-1,-1],[], []]




def point_extension(p1, p2):
  p3 = [0,0,0]
  for i in range(len(p1)):
    p3[i] = (p2[i] +10*(p2[i]-p1[i]))
  return p3
