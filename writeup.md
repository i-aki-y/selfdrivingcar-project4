
# Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to the center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration_input_with_corners.png "Corner Detection"
[image2]: ./output_images/calibration_result.png "Calibration Result"
[image3]: ./output_images/undistort.png "Undistort Example"
[image4]: ./output_images/warp_explain.png "Warp Example"
[image5]: ./output_images/color_space_comparison.png "Color Spaces Examples"
[image6]: ./output_images/combined_colorspace_binarization.png "Combined Binarization"
[image7]: ./output_images/histogram_of_binarized_image.png "Binarized Image Histogram"
[image8]: ./output_images/lane_pixels_with_boxes.png "Fitting Result"
[image9]: ./output_images/lane_pixels_with_search_area.png "Fitting Result"
[image10]: ./output_images/process_image_example.png "Output Example"
[image11]: ./output_images/challange_example.png "Challange Example"
[video1]: ./project_video.mp4 "Video"


## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

I performed camera calibration in the notebook `./Calibration.ipynb`. In order to perform camera calibration, 20 checkarboard images which taken from different angles are prepared.

The `cv2.calibrateCamera()` function in OpenCV calculates the camera matrix `mtx` and distortion coefficients `dist` from the relation between world coordinate of checkerboard corners and coordinates of checkboard corners in the given image.

We can assume that the world coordinate of checkerboard corners are located on the $x-y$ plane at $z=0$. In my code, `objp` is defined as a variable which contains the world coordinate of checkerboard corners. I defined the coordinate are `(0, 0, 0), (0, 1, 0), (0, 2, 0), ..., (5, 7, 0), (5, 8, 0)`, since our checkerboard has 9x6 inner corners.

Next, we should determine the coordinate of projection of checkerboard   corners. The `cv2.findChessboardCorners()` function in OpenCV gives positions of checkerboard image. In my code, the prepared images are converted to grayscale and applied to the function. I stored the resulting corner potions in the `img_points` variable, while the corresponding world coordinates `objp` is appended to the `obj_points` variable.

```py
img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

if ret:
    ## add calibration points when the corners are detected.
    obj_points.append(objp)
    img_points.append(corners)
```

Some images misses some corners and the `cv2.findChessboardCorners()` function fails to detect the positions. In such case, I simply ignored the result. The found positions of projection images are shown in Figure 1.

![FoundCorner][image1]
Figure 1: The result of `cv2.findChessboardCorners()`. Some image failed to find corners position since the images have missing parts.

By applying the `obj_points` and `img_points` to the `cv2.calibrateCamera()` I obtained the `mtx` and `dist`.

```py
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, img_size,None,None)
```

To get a calibrated test image, I applied a test image to the function `cv2.undistort()` by using the obtained `mtx` and `dist` as parameters.


```py
dst = cv2.undistort(img, mtx, dist, None, mtx)
```

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate camera calibration result. I will show the result of undistort function of checkerboard test image and road example image in Figure 2.

![CalibrationResult][image2]
![UndistortExample][image3]
Figure 2: An example of calibration results is shown. Upper left and right is checkerboard test image and lower images are road images. The original images are shown in left side, while undistorted versions are right.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of multiple color thresholds to generate a binary image. The code is written in the function `combined_thres()` in `utils.py` (line:160). In order to determine effective color spaces, I compared the image of each channel in different color spaces (see, "3. Binalization" section in `Analysis.ipynb`). Figure 3 depicts the examples.

![ColorSpaces][image5]
Figure 3: Images with different color space channels are shown. Each columns correspond to different test images ("test1.jpg", "test3.jpg", "test4.jpg", "test5.jpg"), while rows are different color channels ("gray", "rgb_r", "rgb_g", "rgb_b","hls_h", "hls_l", "hls_s", "hsv_h", "hsv_s", "hsv_v", "lab_l", "lab_a", "lab_b") from top to bottom.

Finally, I selected the combination of the following color thresholds. I also examined the gradient threshold, but I did not employ the gradient threshold in this solution. Examples of a combined binarization with "LAB B" are shown in Figure 4.

| color space | channel | thresholds |
|-------------|---------|------------|
| LAB         | B       | (150, 255) |
| HLS         | L       | (220, 255) |
| HSV         | V       | (230, 255) |


![Binarization][image6]
Figure 4: The result of combined binarization of different test images are shown.

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

I defined `make_warp_matrix()`, `warp_img()` functions in `utils.py` (line:19 and 44).
The `make_warp_matrix()` calculate a transformation matrix and its inverse version from given `src_points` which defines a region of original image. In the `make_warp_matrix()` function, destination points are defined from image size and offset values("utils.py, line:28-32").

```py
xoffset = 300
yoffset = 100

y_top, y_bottom = yoffset, img_size[1]
x_left, x_right = xoffset, img_size[0] - xoffset
```

The `warp_img()` function is a simple wrapper function which passes given image and transformation matrix to the `cv2.warpPerspective()` function.

An example of perspective transformation is shown in Figure 5. The trapezoid lines with red color overlaid in an original image are mapped to the rectangle of the transformed image. The relation between source points and destination points are followings:

|  Source   | Destination |
|:---------:|:-----------:|
| 595,  450 |  300, 100   |
| 685,  450 |  980, 100   |
| 1115, 720 |  980, 720   |
| 195,  720 |  300, 720   |


![Perspective transformation result][image4]
Figure 5: An example of perspective transformation. Left is an original test image. Right is the result of the transformation.

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

According to the lecture video, I identified lane-line pixels by using 2nd order polynomial fitting. In the function `fit_lane()` (`utils.py`, line: 307), I determined the $x$ position of the lanes in the bottom of the image. I assumed that left and right lane-lines are located in left and right half of image. The peaks of histogram calculated from the bottom area of the binarized image are employed as candidates of $x$ position of the left and right lanes.

![Binarized image and its histogram][image7]
Figure 6: Relation between a histogram and binarized image.

The area around the peaks is considered as a lane pixels. I introduced small rectangular regions in which the pixels are considered as a lane area and inferred the position of the regions. After determination of lane pixels, I apply 2nd order polynomial fitting for each lane pixels.

![Fitting result][image8]
Figure 7: Prediction of the lane pixels and fitting result.

Once we obtained a fitting curve, I can use the curve in order to update the fitting result. The function `update_fit_lane()` (`utils.py`, line: 398) updates the curves based on the previous fitting curve. The updated curve and search area the region where the pixels are considered as lane are depicted in Figure 8.

![Fitting update result][image9]
Figure 8: Fitting result and search region after update process.


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to the center.

Since we got a polynomial expression of the lanes, we can obtain the radius of curvature from the following formula.

$$
R_{curve} = \frac{ [1 + (\frac{dx}{dy})^2]^{3/2} }{| \frac{d^2x}{dy^2}|}
$$

I calculate the curvature in the `measure_curvature_and_position()` method of `Lane class` (`utils.py`, line: 267).

I also calculate the lane position in the code line 261-265. In order to calculate the lane center, I average left and right lane positions. With the assumption of that the vehicle position is the center of the image, vehicle position from the lane center is given by the following code (`process_image()` in `AdvanLaneFinding.ipynb `).

```py
vihecle_center = (img_detected.shape[1]/2) * xm_per_pix
lane_center = (left_line.line_pos + right_line.line_pos) / 2.0
distance_from_center = vihecle_center - lane_center
```

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in `process_image()` function in `AdvanLaneFinding.ipynb`. Here is an example of my result on a test image:

![Solution example][image10]
Figure 11: An example of my solution.

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here"s a [link to my video result](./output_video/project_solution.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I applied this pipeline to the `challange_video.mp4` and the got the poor result ([link to challenge result](./output_video/challenge_solution.mp4)). In this case, my solution fails to detect the white lanes in the binarized images. This means that my threshold tuning is overfitted to the project_video.mp4 examples. One approach to fix this problem is to collect more examples and perform further tuning.

![alt text][image11]
Figure 12: An example of challenge video result. In this case, only the yellow lane is detected and detection of right lane completely failed.
