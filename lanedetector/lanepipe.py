#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 23:37:13 2017

@author: srikanthnarayanan

This module contains the definition of optimumn pipeline designed for lane
detection video
"""

import cv2
import numpy as np
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
        self.nwindows = window
        self.margin = margin
        self.minpix = minpix

        # Create Left and Right Lane Objects
        self.Left_Lane = Line(lanetype="Left")
        self.Right_Lane = Line(lanetype="Right")

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

        print("Lane Detector : " + "Camera Calibration Completed")

    def undistort(self):
        '''
        A method to undistort an image using camera calibration
        '''
        self.image_shape = (self.image.shape[1], self.image.shape[0])
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.objpoints, self.imgpoints, self.image_shape, None, None)
        undist = cv2.undistort(self.image, mtx, dist, None, mtx)
        
        return undist

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

    def _getHLS(self, img, thresh):
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

    def _getLUV(self, img, thresh):
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

    def _getLab(self, img, thresh):
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

    def runpipeline(self, image):
        '''
        Method defintion to run pipeline by video processing workflows
        '''
        # Undistort Image
        self.image = image
        self.undist_img = self.undistort()

        # Perform Perspective Transform
        self.get_perspective()

        # Identify lanes using binary threhold
        self.get_binary_threhold()

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = self.combined_binary.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        if self.Left_Lane.detected:  # Search from previous found lane
            leftx, lefty, self.Left_Lane.detected = \
                self.Left_Lane.searchfromexisting(nonzerox, nonzeroy,
                                                  self.nwindows,
                                                  self.combined_binary,
                                                  self.margin, self.minpix)
        if self.Right_Lane.detected:  # Search from previous found lane
            rightx, righty, self.Right_Lane.detected = \
                self.Right_Lane.searchfromexisting(nonzerox, nonzeroy,
                                                   self.nwindows,
                                                   self.combined_binary,
                                                   self.margin, self.minpix)
        if not self.Left_Lane.detected:  # Peform Full Search
            leftx, lefty, self.Left_Lane.detected = \
                self.Left_Lane.fulllanesearch(nonzerox, nonzeroy,
                                              self.nwindows,
                                              self.combined_binary,
                                              self.margin, self.minpix)
        if not self.Right_Lane.detected:  # Peform Full Search
            rightx, righty, self.Right_Lane.detected = \
                self.Right_Lane.fulllanesearch(nonzerox, nonzeroy,
                                               self.nwindows,
                                               self.combined_binary,
                                               self.margin, self.minpix)
        # Update Left and Right Lane Object
        self.Left_Lane.x = leftx
        self.Left_Lane.y = lefty
        self.Right_Lane.x = rightx
        self.Right_Lane.y = righty
        
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        ################### Left #####################
        # Smooth of Polynomials
        self.Left_Lane.fit_coeff0.append(left_fit[0])
        self.Left_Lane.fit_coeff1.append(left_fit[1])
        self.Left_Lane.fit_coeff2.append(left_fit[2])
        left_fit = [np.mean(self.Left_Lane.fit_coeff0),
                    np.mean(self.Left_Lane.fit_coeff1),
                    np.mean(self.Left_Lane.fit_coeff2)]

        # get left fit x values
        ploty = np.linspace(0, self.combined_binary.shape[0]-1,
                            self.combined_binary.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        self.Left_Lane.fitx = left_fitx[::-1]

        ################### Right #####################
        # Smooth of Polynomials
        self.Right_Lane.fit_coeff0.append(right_fit[0])
        self.Right_Lane.fit_coeff1.append(right_fit[1])
        self.Right_Lane.fit_coeff2.append(right_fit[2])
        right_fit = [np.mean(self.Right_Lane.fit_coeff0),
                     np.mean(self.Right_Lane.fit_coeff1),
                     np.mean(self.Right_Lane.fit_coeff2)]

        # get Right fit x values
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        self.Right_Lane.fitx = right_fitx[::-1]

        # Calculate radius of Curvature
        left_radius = self.Left_Lane.getradius(ploty)
        right_radius = self.Right_Lane.getradius(ploty)

        avg_radius = np.int((left_radius + right_radius) / 2)

        # Radius update every 5 frames
        if self.Left_Lane.framecount % 5 == 0:
            self.Left_Lane.radius = left_radius
            self.Right_Lane.radius = right_radius

        # Get vehicle position
        # Assumption camera in the center
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        lmean = np.mean(left_fitx)
        rmean = np.mean(right_fitx)
        lane_center = np.mean([lmean, rmean])
        img_center = self.combined_binary.shape[1] / 2
        vehicle_pos = (img_center - lane_center) * xm_per_pix

        # Peform Inverse Perspective Transform
        Minv = cv2.getPerspectiveTransform(self.dst, self.src)

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(self.combined_binary).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        trans_left = [np.transpose(np.vstack([self.Left_Lane.fitx, ploty]))]
        pts_left = np.array(trans_left)
        trans_right = [np.flipud(np.transpose(np.vstack([self.Right_Lane.fitx,
                                                         ploty])))]
        pts_right = np.array(trans_right)
        pts = np.hstack((pts_left, pts_right))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        # Warp the blank back to original image space using
        # inverse perspective matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, Minv,
                                      (self.combined_binary.shape[1],
                                       self.combined_binary.shape[0]))
        # Combine the result with the original image
        result = cv2.addWeighted(self.undist_img, 1, newwarp, 0.5, 0)

        # print vehicle postion
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(result, 'Vehicle center : {:.2f}m'.format(vehicle_pos), \
                    (100, 80), font, 2, (255,255,255), thickness=2)
        # Print Radius
        cv2.putText(result, 'Lane Curvature : {}m'.
                    format(avg_radius), (100, 120), font, 2, (255,255,255),
                    thickness=2)
        self.Left_Lane.framecount += 1
        self.Right_Lane.framecount += 1

        return result


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
        self.fitx = None

        # Count Frames
        self.framecount = 0

    def fulllanesearch(self, nonzerox, nonzeroy, nwindows, binary_warped,
                       margin, minpix):
        '''
        Method to peform a full lane search using the sliding window
        method
        '''
        # Take a histogram of the bottom half of the image
        bottom_half = np.int(binary_warped.shape[0]/2)
        histogram = np.sum(binary_warped[bottom_half:, :], axis=0)
        self.test_hist = histogram

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

    def searchfromexisting(self, nonzerox, nonzeroy, nwindows,
                           binary_warped, margin, minpix):
        '''
        Method to find lanes form exsisting lane postions
        '''
        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)

        # Create empty lists to receive left and right lane pixel indices
        lane_inds = []

        # Set X Current Status
        X_Current_Stat = True

        # Step through the windows one by one
        for window in range(nwindows):
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            y_mean = np.mean([win_y_low, win_y_high])
            if X_Current_Stat:
                x_current = (np.mean(self.fit_coeff0))*y_mean**2 + \
                            (np.mean(self.fit_coeff1))*y_mean + \
                            (np.mean(self.fit_coeff2))
            win_x_low = x_current - margin
            win_x_high = x_current + margin

            # Identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                         (nonzerox >= win_x_low) &
                         (nonzerox < win_x_high)).nonzero()[0]

            # Append these indices to the lists
            lane_inds.append(good_inds)

            # If found > minpix, recenter next window to mean position
            # Dont update x_current
            if len(good_inds) > minpix:
                X_Current_Stat = False

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
            self.detected = False

        return x, y, self.detected

    def getradius(self, ploty):
        '''
        method to get radius of curvature
        '''
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        fit_cr = np.polyfit(ploty*ym_per_pix, self.fitx*xm_per_pix, 2)
        y_eval = np.max(ploty)
        curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix +
                          fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
        return curverad

if __name__ == "__main__":
    pass
    