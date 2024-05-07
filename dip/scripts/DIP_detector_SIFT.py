#!/usr/bin/env python3 

import sys
import os
import argparse
import cv2
import rospy
import roslib
from std_msgs.msg import String
from sensor_msgs.msg import Image, CompressedImage
from threading import Lock
from cv_bridge import CvBridge, CvBridgeError
import message_filters

# local libraries and msgs
# from rightHandedAngleLocatorStereo import detect_object
from triangle_canny import detect_object

# from bboxTracker import BoxTrackerKF
from target_following.msg import TargetObservation, TargetObservations





class PipelineCanny:
    """ 
    Class for detecting diver (front right camera) and publish target bounding box 
    """
    def __init__(self, real_time=False):
        # self.objDetect = detect_object()
        # self.drTracker = BoxTrackerKF()
        # # target BBox center
        self.temp_BBox = []
        self.target_x, self.target_y = None, None
        self.mutex = Lock()

        self.bench_test, self.publish_image = False, True
        # settings only for real-time testing
        if real_time:
            print("running")
            rospy.init_node('follower', anonymous=True)
            #read_cameras()
            self.bridge = CvBridge()
            # self.topic_right = rospy.get_param('/right_object_detectorimage/camera_topic')
            # self.topic_left = rospy.get_param('/left_object_detector/camera_topic')
            '''For using 2 cameras'''
            imageL = message_filters.Subscriber("/loco_cams/right/image_raw", Image)
            imageR = message_filters.Subscriber("/loco_cams/right/image_raw", Image)

            # Synchronize images
            ts = message_filters.ApproximateTimeSynchronizer([imageL, imageR], queue_size=2, slop=1)
            ts.registerCallback(self.image_callback)
            # '''For 1 camera'''
            # image_right = rospy.Subscriber("/loco_cams/right/image_raw", Image, self.imageCallBackright, queue_size=3)
            # image_left = rospy.Subscriber(self.topic_left, Image, self.imageCallBack, queue_size=3)
            # Bbox publisher and processed output image publisher
            self.bbox_pub  = rospy.Publisher("/target/observation", TargetObservations, queue_size=1)
            self.ProcessedRaw = rospy.Publisher('/object_follower/image', Image, queue_size=1)
            try:
                rospy.spin()
            except KeyboardInterrupt:
                print("Rospy Sping Shut down")

    

    def image_callback(self, imageL, imageR):
        br = CvBridge()
        rospy.loginfo("receiving frame")
        self.imageLeft = br.imgmsg_to_cv2(imageL, "bgr8")
        self.imageRight = br.imgmsg_to_cv2(imageR, "bgr8")	
        if self.imageLeft is None:
            print ('frame dropped, skipping tracking')
        else:
            self.ImageProcessor()

    def imageCallBackright(self, r_im):
        """ 
        CallBack function to get the image (from back camera) through the 
        ros-opencv-bridge and start processing
        """
        try:
            self.original = self.bridge.imgmsg_to_cv2(r_im, "bgr8")
        except CvBridgeError as e:
            print(e)
        if self.original is None:
            print ('frame dropped, skipping tracking')
        else:
            self.ImageProcessor()

    def ImageProcessor(self):
        """ 
        Process each frame
        > Detect Pointing location
        > Publish object bounding box
        """
        # object detection
        sift_bbox, pose, triangle = detect_object(self.imageLeft, self.imageRight)
        n_, m_, _ = self.imageLeft.shape
        
        if sift_bbox == [-1,-1,-1,-1] and self.temp_BBox == []:
            rospy.loginfo("No pointing/Object")
            pass 
        
        
        elif self.temp_BBox != []:
            '''elif Statement should be replaced with other bounding box tracking method'''
            
            rospy.loginfo('Tracking from previous')
            sift_bbox, pose, triangle = detect_object(self.imageLeft, self.imageRight)
            n_, m_, _ = self.imageLeft.shape
            
            # rospy.loginfo(BBox)
            if sift_bbox == [-1,-1,-1,-1]:
                rospy.loginfo("No new pointing/ object")
                top_left_x, top_left_y, width, height = self.temp_BBox
                self.mutex.acquire()
                # the bbox msg --------------------------
                obs = TargetObservation()
                obs.header.stamp = rospy.Time.now()
                mobs = TargetObservations()
                mobs.header.stamp = obs.header.stamp
                obs.target_visible = True
                obs.top_left_x = top_left_x
                obs.top_left_y = top_left_y
                obs.width = (width)
                obs.height = (height)
                obs.image_width = m_
                obs.image_height = n_
                obs.class_prob = 1.0
                obs.class_name = 'object'
                mobs.observations.append(obs)
                #--------------------------
                self.mutex.release()
                self.bbox_pub.publish(mobs)


                if self.publish_image:
                    rospy.loginfo ("Tracking >> Bbox: {0}".format(self.temp_BBox))
                    cv2.rectangle(self.imageLeft, (top_left_x, top_left_y), (top_left_x+width, top_left_y+height), (0, 255, 255), 2)
                    msg_frame = CvBridge().cv2_to_imgmsg(self.imageLeft, encoding="bgr8")
                    self.ProcessedRaw.publish(msg_frame)

            elif sift_bbox != [-1,-1,-1,-1]:
                rospy.loginfo("New Object found")
                top_left_x, top_left_y, width, height = sift_bbox
                # prepare and publish target bounding box
                self.mutex.acquire()
                # the bbox msg --------------------------
                obs = TargetObservation()
                obs.header.stamp = rospy.Time.now()
                mobs = TargetObservations()
                mobs.header.stamp = obs.header.stamp
                obs.target_visible = True
                obs.top_left_x = top_left_x
                obs.top_left_y = top_left_y
                obs.width = (width)
                obs.height = (height)
                obs.image_width = m_
                obs.image_height = n_
                obs.class_prob = 1.0
                obs.class_name = 'object'
                mobs.observations.append(obs)
                #--------------------------
                self.mutex.release()
                self.bbox_pub.publish(mobs)
                self.temp_BBox = sift_bbox
                # rospy.loginfo(self.temp_BBox)
                
                if self.publish_image:
                    rospy.loginfo ("Object detected >> Bbox: {0}".format(sift_bbox))
                    for pts in pose:
                        cv2.circle(self.imageLeft, (int(pts[0]), int(pts[1])), 3, (0,255,0), -1)
                    cv2.rectangle(self.imageLeft, (sift_bbox[0], sift_bbox[1]), (sift_bbox[0] + sift_bbox[2], sift_bbox[1] + sift_bbox[3]), (36,255,12), 2)
                    cv2.polylines(self.imageLeft, pts=[triangle], color=(255, 255, 255), isClosed=True)
                    msg_frame = CvBridge().cv2_to_imgmsg(self.imageLeft, encoding="bgr8")
                    self.ProcessedRaw.publish(msg_frame)
            '''End of elif stantement to be replaced by tracker'''

        elif sift_bbox != [-1,-1,-1,-1]:
            rospy.loginfo("Object found")
            top_left_x, top_left_y, width, height = sift_bbox
            # Publish target bounding box
            self.mutex.acquire()
            # the bbox msg --------------------------
            obs = TargetObservation()
            obs.header.stamp = rospy.Time.now()
            mobs = TargetObservations()
            mobs.header.stamp = obs.header.stamp
            obs.target_visible = True
            obs.top_left_x = top_left_x
            obs.top_left_y = top_left_y
            obs.width = (width)
            obs.height = (height)
            obs.image_width = m_
            obs.image_height = n_
            obs.class_prob = 1.0
            obs.class_name = 'object'
            mobs.observations.append(obs)
            #--------------------------
            self.mutex.release()
            self.bbox_pub.publish(mobs)
            self.temp_BBox = sift_bbox
            rospy.loginfo(self.temp_BBox)
            
            if self.publish_image:
                rospy.loginfo("Object detected >> Bbox: {0}".format(sift_bbox))
                for pts in pose:
                    cv2.circle(self.imageLeft, (int(pts[0]), int(pts[1])), 3, (0,255,0), -1)
                cv2.rectangle(self.imageLeft, (sift_bbox[0], sift_bbox[1]), (sift_bbox[0] + sift_bbox[2], sift_bbox[1] + sift_bbox[3]), (36,255,12), 2)
                cv2.polylines(self.imageLeft, pts=[triangle], color=(255, 255, 255), isClosed=True)
                msg_frame = CvBridge().cv2_to_imgmsg(self.imageLeft, encoding="bgr8")
                self.ProcessedRaw.publish(msg_frame)


go = PipelineCanny(real_time=True)

'''
    ##########################################################################
    ###   For bench testing with dataset images ###############################
    def showFrame(self, frame, name):
    cv2.imshow(name, frame)
    cv2.waitKey(20)

    # stream images from directory Dir_
    def image_streamimg(self, Dir_):
    from eval_utils import filter_dir
    dirFiles = filter_dir(os.listdir(Dir_))
    for filename in dirFiles:
    self.original = cv2.imread(Dir_+filename)
    self.ImageProcessor()
####################################################################################
'''