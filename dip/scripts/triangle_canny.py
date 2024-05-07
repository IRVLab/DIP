#!/usr/bin/env python3  

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
  # rospy.loginfo(left_points2d_array)
  
  # Half base of Triangle 
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
    # half_base = 100 
   
    triangle = np.array([[mean_endpoint[0],mean_endpoint[1]+half_base],[mean_endpoint[0],mean_endpoint[1]-half_base],[left_points2d_array[3][0]-5,left_points2d_array[3][1]+5]],dtype=np.int32)
    cv2.fillPoly(mask, pts=[triangle], color=(255, 255, 255))
      
    # Canny detection
    gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    masked_left_image = cv2.bitwise_and(gray,mask)
    canny = cv2.Canny(gray, 120, 255, 1)
    # Find contours
    cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Iterate thorugh contours and draw rectangles around contours
    canny_bbox = [-1,-1,-1,-1]
    for c in cnts:
      M = cv2.moments(c)
      if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        state = insideTriangle(triangle, cx, cy) #Check if contour inside area of interest
        if state == True:

          x,y,w,h = cv2.boundingRect(c)
          if x < 0:
              x = 0
          canny_bbox =[x,y,w,h]
          cv2.rectangle(masked_left_image, (x, y), (x + w, y + h), (36,255,12), 2)
    canny_bbox = canny_bbox
      # cv2.imshow('canny', canny)
      # cv2.imshow('image', masked_left_image)
      
     
    return [canny_bbox,left_points2d_array, triangle]
    
    
  elif len(left_points2d_array) != 0:
    #Get mediapipe landmarks and reshape for left image
    shape2d,m = left_points2d_array.shape

    #extend pointing in 3d
    '''3d line  extension elbow and wrist stuff'''
    
    # Left image info
    left_endpoint = point_extension(left_points2d_array[2],left_points2d_array[3])

    '''Find triangle from wrist'''
    #Mean endpoint gives height 
    # triangle base length: 1/2 is distance from endpint 10
    # 
    half_base = 20 

    triangle = np.array([[left_endpoint[0],left_endpoint[1]+half_base],[left_endpoint[0],left_endpoint[1]-half_base],[left_points2d_array[3][0]-5,left_points2d_array[3][1]+5]],dtype=np.int32)

    cv2.fillPoly(mask, pts=[triangle], color=(255, 255, 255))
    
    # Canny detection
    gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    masked_left_image = cv2.bitwise_and(gray,mask)
    canny = cv2.Canny(gray, 120, 255, 1)
    # Find contours
    cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Iterate thorugh contours and draw rectangles around contours
    canny_bbox = [-1,-1,-1,-1]
    for c in cnts:
      M = cv2.moments(c)
      if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        state = insideTriangle(triangle, cx, cy)
        if state == True:

          x,y,w,h = cv2.boundingRect(c)
          if x < 0:
              x = 0
          canny_bbox =[x,y,w,h]
          cv2.rectangle(masked_left_image, (x, y), (x + w, y + h), (36,255,12), 2)
    canny_bbox = canny_bbox
      # cv2.imshow('canny', canny)
      # cv2.imshow('image', masked_left_image)
      
     
    return [canny_bbox,left_points2d_array, triangle]
  
  elif len(right_points2d_array) != 0:
    #Get mediapipe landmarks and reshape for left image
    #Get Right image info
    shape2d,m = right_points2d_array.shape
      
   
    # right image info
    right_endpoint = point_extension(right_points2d_array[2],right_points2d_array[3])

    #find pointing direction
    '''3d line  extension elbow and wrist stuff'''
    
    '''Find triangle from wrist'''
    #Mean endpoint gives height 
    # triangle base length: 1/2 is distance from endpint 10
    half_base = 100 

    triangle = np.array([[right_endpoint[0],right_endpoint[1]+half_base],[right_endpoint[0],right_endpoint[1]-half_base],[right_points2d_array[3][0]-5,right_points2d_array[3][1]+5]],dtype=np.int32)

    
    cv2.fillPoly(mask, pts=[triangle], color=(255, 255, 255))
   # Canny detection
    gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    masked_left_image = cv2.bitwise_and(gray,mask)
    canny = cv2.Canny(gray, 120, 255, 1)
    # Find contours
    cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Iterate through contour centroids and draw rectangles around contours
    canny_bbox = [-1,-1,-1,-1]
    for c in cnts:
      M = cv2.moments(c)
      if M['m00'] != 0:
        cx = int(M['m10']/M['m00']) 
        cy = int(M['m01']/M['m00'])
        state = insideTriangle(triangle, cx, cy)
        if state == True: # If inside draw rectangle

          x,y,w,h = cv2.boundingRect(c)
          if x < 0:
              x = 0
          canny_bbox =[x,y,w,h]
          cv2.rectangle(masked_left_image, (x, y), (x + w, y + h), (36,255,12), 2)
    canny_bbox = canny_bbox
      # cv2.imshow('canny', canny)
      # cv2.imshow('image', masked_left_image)
      
     
    return [canny_bbox,left_points2d_array, triangle] 

  else:
      return [[-1,-1,-1,-1],[], []]



''' Triangle area method esed to determine if centroid of object is within the object of interest:
'''
def insideTriangle(triangle, x, y):
    # Calculate areas of triangles with point and two vertices 
    t_1 = triangle_area(x, y, triangle[1][0],triangle[1][1],triangle[2][0],triangle[2][1])
    t_2 = triangle_area(triangle[0][0],triangle[0][1], x, y, triangle[2][0],triangle[2][1])
    t_3 = triangle_area(triangle[0][0],triangle[0][1], triangle[1][0],triangle[1][1], x, y)
    # Calculate area of full triangle
    A = triangle_area(triangle[0][0],triangle[0][1],triangle[1][0],triangle[1][1],triangle[2][0],triangle[2][1])

    if(A == t_1 + t_2 + t_3):
        return True
    else:
        return False

def triangle_area(x0, y0, x1, y1, x2, y2):
    area = abs((x0 * (y1 - y2) + x1 * (y2 - y0)
                + x2 * (y0 - y1)) / 2)
    return area

def point_extension(p1, p2):
  p3 = [0,0,0]
  for i in range(len(p1)):
    p3[i] = (p2[i] +10*(p2[i]-p1[i]))
  return p3
