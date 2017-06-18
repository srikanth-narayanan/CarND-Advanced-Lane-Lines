[//]: # (Image References)

[image1]: ./output_images/UNDIST_Image.png "Undistorted Image"
[image2]: ./output_images/PERSPECTIVE_image.png "Perspective Transform"
[image3]: ./output_images/L_CHANNEL_binary.png "L_CHANNEL_binary"
[image4]: ./output_images/B_CHANNEL_binary.png "B_CHANNEL_binary"
[image5]: ./output_images/COMBI_CHANNEL_binary.png "Undistorted Image"
[image6]: ./output_images/WINDOW_poly.png "Polynomial Fit"
[image7]: ./output_images/DETECTED.png "Detected Lanes, Curvature and Vehicle Position"
[image8]: ./output_images/DETECTED.png "Detected Lanes, Curvature and Vehicle Position"
[image9]: ./output_images/camera_calib.png "Camera Calibration"

## Advanced Lane Finding Project

This project involves building a image processing pipeline to detect road lanes form a center dash caemra of a vehicle. In order to sucessfully detect the road lanes, the following steps are performed

- Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
- Apply a distortion correction to raw images.
- Apply a perspective transform to rectify binary image ("Top view of the road").
- Use color transforms, to create a thresholded binary image.
- Detect lane pixels and fit to find the lane boundary.
- Determine the curvature of the lane and vehicle position with respect to center.
- Warp the detected lane boundaries back onto the original image.
- Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

## Organisation of the code

The juypter notebook "Advanced_Lane_Finding.ipynb" contains the implementation of building a pipeline for an image. This pipeline is transformed to python package "lanedetector" that supports implementing the pipeline on video files. The "lanedetector" package contains class defintion for laneDetector and Line.

### Find Chessboard Corners

Open CV image read function reads images in the RGB colour space as BGR numpy array. Open CV provides a function `cv2.findChessboardCorners()`, this function identifies the chessboard corners using the pin hole camera model. A `cv2.drawChessboardCorners()` provide a visual indetifier for all the corners detected by the camera. The given chessboard images contains a 9 x 6 matrix.

The code lines are found in cells 2,3,4,5 and 7

Object points that represent the (x, y, z) co-ordinates in the real world, with z axis contains 0 length. the image points contains the array of detected corners.

![Chessboard Corners][image9]

### Undistort Image

The `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function. The code lines are found in cells 8, 9 and 10.

![Undistort Image][image1]

### Perspective Transform

A perspective transform is applied in order to generate a top view a.k.a birds eye view of the road lanes. The region of interest for this pipeline are the vehicle lanes. Hence a specific region of the lanes are identified as source (`src`) and these are warped in to a destination (`dst`) region. the following are the source and destination of choice

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 490, 480      | 0, 0          | 
| 810, 480      | 1280, 0       |
| 1250, 720     | 1250, 720     |
| 100, 720      | 100, 720      |

The open cv functions `cv2.getPerspectiveTransform` is used to generate the perpective transform and `cv2.warpPerspective` is used to generate the warped image. All these code are built in a function called `getperspective()`. The code lines are in cells 11, 12 and 13.

![Perspective transformed][image2]

### Colour transform and binary threshold

The pixels associated with lane lines have to be isolated to detect the lanes better





#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 1 through 8 in the file `example.py` (output_images/examples/example.py) (or, for example, in the 3rd code cell of the IPython notebook).  The `warper()` function takes as inputs an image (`img`), as well as source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 55, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 10), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 55), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 585, 460      | 320, 0        | 
| 203, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 695, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
