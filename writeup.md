
# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration_input_with_corners.png "Corner Detection"
[image2]: ./output_images/calibration_result.png "Calibration Result"
[image3]: ./output_images/undistort.png "Undistort Example"
[image4]: ./output_images/perspective_explain.png "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I performed camera calibration in the notebook "./Calibration.ipynb". In order to perform camera calibration, 20 checkarboard images which taken from different angles are prepared.

The `cv2.calibrateCamera()` function in OpenCV calculates the camera matrix `mtx` and distortion coefficients `dist` from the relation between world coordinate of target objects which are checkarboard corners in this case and coordinate of projection of the checkarboard which are position of checkboard corners in the given image.

We can assume that the world coordinate of checkarboard corners are located on the $x-y$ plane at $z=0$. In my code, `objp` is defined as a variable which contains the world coordinate of checkearboard corners. I defined the coordinate are `(0, 0, 0), (0, 1, 0), (0, 2, 0), ..., (5, 7, 0), (5, 8, 0)`, since our checkarboard has 9x6 inner corners.

Next, we should determine the coordinate of projection of checkarboard   corners. The `cv2.findChessboardCorners()` function in OpenCV gives positions of checkarboard image. In my code, the prepared images are converted to grayscale and applied to the function. I stored the resulting corner potions in the `img_points` variable, while the corresponding world coordinates `objp` is appended to the `obj_points` variable.

```py
img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

if ret:
    ## add calibration points when the corners are detected.
    obj_points.append(objp)
    img_points.append(corners)
```

Some images misses parts of corners and the `cv2.findChessboardCorners()` function fails to detect the positions. In such case, I simply ignored the result. The found position of projection images are shown in Figure 1.

![FoundCorner][image1]
Figure 1: The result of `cv2.findChessboardCorners()`. Some image failed to find corners position, since the images have missing parts.

By applying the `obj_points` and `img_points` to the `cv2.calibrateCamera()` I obtained the `mtx` and `dist`.

```py
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size,None,None)
```

To get a calibrated test image, I applied a test image to the function `cv2.undistort()` by using the obtained `mtx` and `dist` as a parameters.


```py
dst = cv2.undistort(img, mtx, dist, None, mtx)
```

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate camera calibration result. I will show the result of undistort function of checkarboard test image and road example image in Figure 2.

![CalibrationResult][image2]
![UndistortExample][image3]
Figure 2: An example of calibration results is shown. Upper left and right are checkarboard test image and lower images are road images. The original images are shown in left side, while undistorted versions are right.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I defined `make_warp_matrix()`, `warp_img()` functions in `utils.py` file.
The `make_warp_matrix()` calculate a perspective transform matrix and its inverse version from given `src_points` which defines a region of original image.

![alt text][image4]
Fiture 4:

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
