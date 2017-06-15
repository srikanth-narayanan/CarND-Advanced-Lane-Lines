#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 23:37:13 2017

@author: srikanthnarayanan

This module contains the definition of optimumn pipeline designed for lane
detection video
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
from collections import deque


class laneDetector(object):
    '''
    A class defintion for lane detection pipeline. This class has definition to
    methods and process need to perform lane detection
    '''
    def __init__(self, srcpoint, dstpoint, chessvertical=6, chesshorizontal=9,
                 window=9, margin=100, minpix=50):
        '''
        Constructor to intialise the object variables
        '''
        self.chessver = chessvertical
        self.chesshor = chesshorizontal
        self.src = srcpoint
        self.dst = dstpoint
        self.nwindow = window
        self.margin = margin
        self.minpix = minpix
        pass

    def calibrate_camera(self, imagelist):
        '''
        A method to find the object and image points from a chess board
        calibration image
        '''
        # Arrays to store object and image points from the chessboard images
        self.objpoints = []  # 3d points in real world
        self.imgpoints = []  # 2D points in image plane
        objp = np.zeros((self.chessver * self.chesshor, 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chesshor,
                               0:self.chessver].T.reshape(-1, 2)
        self.chesscorndetect = []
        self.orgchessimg = []
        for idx, img in enumerate(imagelist):
            imgr = cv2.imread(img)
            gray = cv2.cvtColor(imgr, cv2.COLOR_BGR2GRAY)
            # Get chess board corner
            ret, corners = cv2.findChessboardCorners(gray, (self.chesshor,
                                                            self.chessver),
                                                     None)
            # If points are found add to list
            if ret:
                self.orgchessimg.append(imgr.copy())
                self.objpoints.append(objp)
                self.imgpoints.append(corners)
                # Draw the chess board corners
                cv2.drawChessboardCorners(imgr, (self.chesshor, self.chessver),
                                          corners, ret)
                self.chesscorndetect.append(imgr)

    def undistort(self):
        '''
        A method to undistort an image using camera calibration
        '''
        self.image_shape = (self.image.shape[1], self.image.shape[0])
        cam_cal = cv2.calibrateCamera(self.objpoints, self.imgpoints,
                                      self.image_shape, None, None)
        cam_atr = ['ret', 'mtx', 'dist', 'rvecs', 'tvecs']
        [setattr('self', name, val) for name, val in zip(cam_atr, cam_cal)]
        self.undist = cv2.undistort(self.image, self.mtx, self.dist,
                                    None, self.mtx)

    def get_perspective(self):
        '''
        This function performs a perspective transform
        '''
        perspect_mat = cv2.getPerspectiveTransform(self.src, self.dst)
        self.perspect_img = cv2.warpPerspective(self.undist_img, perspect_mat,
                                                self.image_shape)

    def get_binary_threhold(self, labthresh=(165, 250), luvthresh=(224, 255)):
        '''
        This function generates a binary threhold image using lab and luv.
        Lab detects yellow lines better and Luv detects white lines.
        '''
        # get b in Lab
        b_chnl = self._getLab(self.perspect_img, thresh=labthresh)
        l_chnl = self._getLUV(self.perspect_img, thresh=luvthresh)

        # combine binary
        self.combined_binary = np.zeros_like(l_chnl)
        self.combined_binary[(l_chnl == 1) | (b_chnl == 1)] = 1

    def _getHLS(self, img, thresh=(0, 255)):
        '''
        Helper function to get HLS threshold of the image
        '''
        # Convert to HLS Color Space
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        # Apply a threshold to S channel
        s_channel = hls[:, :, 2]
        binary = np.zeros_like(s_channel)
        binary[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
        return binary

    def _getLUV(img, thresh=(0, 255)):
        '''
        Helper function to get LUV threshold of the image
        '''
        # Convert to LUV Color Space
        luv = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
        # Apply a threshold to L channel
        l_channel = luv[:, :, 0]
        binary = np.zeros_like(l_channel)
        binary[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 1
        return binary

    def _getLab(img, thresh=(0, 255)):
        '''
        Helper function to get LUV threshold of the image
        '''
        # Convert to LUV Color Space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # Apply a threshold to L channel
        # b is for blue-yellow (Since we want to detect yellow lines better)
        b_channel = lab[:, :, 2]
        binary = np.zeros_like(b_channel)
        binary[(b_channel > thresh[0]) & (b_channel <= thresh[1])] = 1
        return binary

    def runpipeline(self):
        '''
        Method defintion to run pipeline by video processing workflows
        '''


class Line():
    '''
    This class defines the object properties of a detected line
    '''
    def __init__(self, lanetype="Left"):
        '''
        Constructor object for the class to initialise attributes.
        '''
        # Line type
        self.lanetype = lanetype
        
        # was the line detected in the last iteration
        self.detected = False

        # last found x and y
        self.x = None
        self.y = None

        # Last radius of curvature
        self.radius = None

        # Last 10 polynomial fit for the lanes
        self.fit_coeff0 = deque(maxlen=10)
        self.fit_coeff1 = deque(maxlen=10)
        self.fit_coeff2 = deque(maxlen=10)
        self.fit = None

        # Count Frames
        self.framecount = 0

    def fulllanesearch(self, nonzerox, nonzeroy, binary_warped, nwindows,
                       margin, minpix):
        '''
        Method to peform a full lane search using the sliding window
        method
        '''
        # Take a histogram of the bottom half of the image
        bottom_half = np.int(binary_warped.shape[0]/2)
        histogram = np.sum(binary_warped[bottom_half:, :], axis=0)

        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)

        if self.lanetype == "Left":
            base = np.argmax(histogram[:midpoint])
        elif self.lanetype == "Right":
            base = np.argmax(histogram[midpoint:]) + midpoint
        else:
            raise Exception("Unknown Lane Type")

        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)

        # Current positions to be updated for each window
        x_current = base

        # Create empty lists to receive left and right lane pixel indices
        lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin

            # Identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_x_low) &
                         (nonzerox < win_x_high)).nonzero()[0]

            # Append these indices to the lists
            lane_inds.append(good_inds)

            # If found > minpix pixels, recenter next window to mean position
            if len(good_inds) > minpix:
                x_current = np.int(np.mean(nonzerox[good_inds]))

        # Concatenate the arrays of indices
        lane_inds = np.concatenate(lane_inds)

        # Extract left and right line pixel positions
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]

        # check if values are found
        if np.sum(x) > 0:
            self.detected = True
        else:
            x = self.x
            y = self.y

        return x, y, self.detected

    def serachfromexisting(self, x, y):
        '''
        Method to find lanes form exsisting lane postions
        '''

    def getradius(self):
        '''
        method to get radius of curvature
        '''
        pass

    def getvehicleposition(self):
        '''
        method to get vehicle position with respect to the lane
        '''
        pass
    