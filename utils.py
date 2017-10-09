import pickle
import numpy as np
import cv2
import glob
from pathlib import Path


class EmptyDataError(Exception):
    """Exception raised for errors when empty data

    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


def make_warp_matrix(img, src_points):
    """
    Args:
    img -- Image array
    src_points -- (top_left, top_right, bottom_right, bottom_left) points which were picked up from test image
    """

    img_size = (img.shape[1], img.shape[0])
    src = np.float32(src_points)
    xoffset = 300
    yoffset = 100

    y_top, y_bottom = yoffset, img_size[1]
    x_left, x_right = xoffset, img_size[0] - xoffset

    dst = np.float32([
        [x_left, y_top], [x_right, y_top], [x_right, y_bottom], [x_left, y_bottom]
    ])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    return M, Minv, dst


def warp_img(img, M):
    """
    img -- src image
    M -- matrix calculated by `cv2.getPerspectiveTransform(src, dst)`
    """

    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped


def gray_conv(rgb_img, color_space='gray'):
    """
    convert RGB color to gray scale with the schema specified by color_space.

    ARGS:

    rgb_img -- RGB order image.
    color_space -- string which indicate color space. 'hls_s' means "S" channel of "HLS" color.

    """

    if color_space == 'gray':
        img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    elif color_space == 'rgb_r':
        img = rgb_img[:, :, 0]
    elif color_space == 'rgb_g':
        img = rgb_img[:, :, 1]
    elif color_space == 'rgb_b':
        img = rgb_img[:, :, 2]
    elif color_space == 'hls_h':
        hls = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HLS)
        img = hls[:, :, 0]
    elif color_space == 'hls_l':
        hls = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HLS)
        img = hls[:, :, 1]
    elif color_space == 'hls_s':
        hls = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HLS)
        img = hls[:, :, 2]
    elif color_space == 'hsv_h':
        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        img = hsv[:, :, 0]
    elif color_space == 'hsv_s':
        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        img = hsv[:, :, 1]
    elif color_space == 'hsv_v':
        hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        img = hsv[:, :, 2]
    elif color_space == 'lab_l':
        lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
        img = lab[:, :, 0]
    elif color_space == 'lab_a':
        lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
        img = lab[:, :, 1]
    elif color_space == 'lab_b':
        lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
        img = lab[:, :, 2]
    else:
        raise ValueError("{0} is unknown color_space".format(color_space))
    return img


def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255), color_space='gray'):
    # Calculate directional gradient
    # Apply threshold
    gray = gray_conv(img, color_space=color_space)
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    scale_sobel = np.uint8(255*abs_sobel/abs_sobel.max())
    grad_binary = np.zeros_like(scale_sobel)
    grad_binary[(scale_sobel >= thresh[0]) & (scale_sobel <= thresh[1])] = 1
    return grad_binary


def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255), color_space='gray'):
    # Calculate gradient magnitude
    # Apply threshold
    gray = gray_conv(img, color_space=color_space)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.sqrt(sobelx**2 + sobely**2)
    scale_sobel = np.uint8(255*abs_sobel/abs_sobel.max())
    mag_binary = np.zeros_like(scale_sobel)
    mag_binary[(scale_sobel >= mag_thresh[0]) & (scale_sobel <= mag_thresh[1])] = 1
    return mag_binary


def dir_thresh(img, sobel_kernel=3, dir_thresh=(0, np.pi/2), color_space='gray'):
    # Calculate gradient direction
    # Apply threshold
    gray = gray_conv(img, color_space=color_space)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    dir_sobel = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    dir_binary = np.zeros_like(dir_sobel)
    dir_binary[(dir_sobel >= dir_thresh[0]) & (dir_sobel <= dir_thresh[1])] = 1

    return dir_binary


def color_channel_thresh(img, thresh=(0, 255), color_space='gray'):
    gray = gray_conv(img, color_space=color_space)
    binary = np.zeros_like(gray)
    binary[(gray >= thresh[0]) & (gray <= thresh[1])] = 1
    return binary


def gray_thresh(gray, thresh=(0, 255)):
    binary = np.zeros_like(gray)
    binary[(gray >= thresh[0]) & (gray <= thresh[1])] = 1
    return binary


def combined_thresh(img):
    params = [("lab_b", 1.0, 150),
              #              ("hls_s", 1.0, 230),
              ("hls_l", 1.0, 220),
              ("hsv_v", 1.0, 230)]
    bins = {}
    res = np.zeros_like(img[:, :, 0])
    for cs, lim, thresh_low in params:
        gray = gray_conv(img, cs)
        gray_clahe = apply_clahe(gray, limit=lim, grid_size=8)
        binary = gray_thresh(gray_clahe, thresh=(thresh_low, 255))
        bins[cs] = binary
        res[(res == 1) | (binary == 1)] = 1

    return res, bins


class Line():

    def __init__(self):
        from collections import deque

        # was the line detected in the last iteration?
        self.detected = False

        # x values of the last n fits of the line
        self.recent_xfitted = deque(maxlen=10)

        # average x values of the fitted line over the last n iterations
        self.bestx = None

        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None

        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]

        # history of polynomial coefficients
        self.fit_history = deque(maxlen=10000)

        # history of polynomial coefficients
        self.best_fit_history = deque(maxlen=10000)

        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        self.radius_of_curvature_history = deque(maxlen=10000)

        # x position of line
        self.line_pos = None

        # history of line posotion from left side of image
        self.line_x_pos_history = deque(maxlen=10000)

        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')

        # history of x values for detected line pixels
        self.detected_x_history = deque(maxlen=10)

        # history of y values for detected line pixels
        self.detected_y_history = deque(maxlen=10)

    def check_fit(self, fit):

        thresh_hold = 0.001

        if len(self.fit_history) < 5:
            return True

        diff = abs(fit[0] - self.best_fit[0])

        if diff > thresh_hold:
            return False
        else:
            return True

    def add_detected(self, x, y):
        self.detected_x_history.appendleft(x)
        self.detected_y_history.appendleft(y)

    def add_fitted_x(self, x):
        self.recent_xfitted.appendleft(x)

    def add_fit(self, fit):
        self.diffs = fit - self.current_fit
        self.current_fit = fit
        self.fit_history.appendleft(fit)

    def measure_curvature_and_position(self, y_eval, xm_per_pix=1, ym_per_pix=1):
        """Measure lane curvature
        Args:
        y_eval -- y value at which the curvature is evaluated
        x -- fitted x value
        y -- fitted y value
        xm_per_pix -- meter per pixel of x axis
        ym_per_pix -- meter per pixel of y axis
        """

        # Fit new polynomials to x,y in world space
        fit = self.best_fit

        # X position
        x_line_position = (fit[0]*y_eval**2 + fit[1]*y_eval*ym_per_pix + fit[2]) * xm_per_pix
        self.line_pos = x_line_position
        self.line_x_pos_history.appendleft(x_line_position)

        # Calculate the new radii of curvature
        curverad = ((1 + (2*fit[0]*y_eval*ym_per_pix + fit[1])**2)**1.5) / np.absolute(2*fit[0])

        self.radius_of_curvature = curverad
        self.radius_of_curvature_history.appendleft(curverad)

        return curverad

    def update_best_fit(self):
        """
        Calculate best fitting result

        Here, weighted mean of recent n fitting result
        is used as best_fitting
        """
        n = 5
        w_decay = 0.5

        recent_fit = np.array(list(self.fit_history)[:n])
        n = recent_fit.shape[0]
        w = np.exp(-w_decay * np.arange(n)).reshape(-1, 1)
        self.best_fit = (recent_fit * w).sum(axis=0)/w.sum()
        self.best_fit_history.appendleft(self.best_fit)

    def update_line(self, x, y, fitted_x, fit):

        if not self.check_fit(fit):
            return False

        self.add_detected(x, y)
        self.add_fitted_x(fitted_x)
        self.add_fit(fit)

        self.update_best_fit()

        return True

    def best_fit_line(self, y):
        return self.best_fit[0]*y**2 + self.best_fit[1]*y + self.best_fit[2]


def fit_lane(binary_warped):
    """ Fit lane curve of given image

    Args:

    binary_warped -- image array which is binarized and warped.
    """

    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                      color=(0, 255, 0), thickness=2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high),
                      color=(0, 255, 0), thickness=2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if not (leftx.any() and lefty.any() and rightx.any() and righty.any()):
        raise EmptyDataError("Empty data")

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return {
        "left_fit": left_fit, "right_fit": right_fit,
        "out_img": out_img,
        "nonzerox": nonzerox, "nonzeroy": nonzeroy,
        "left_lane_inds": left_lane_inds, "right_lane_inds": right_lane_inds,
        "leftx": leftx, "lefty": lefty, "rightx": rightx, "righty": righty,
    }


def update_fit_lane(binary_warped, left_fit, right_fit):

    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
                                                                         left_fit[1]*nonzeroy + left_fit[2] + margin)))

    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
                                                                           right_fit[1]*nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    if not (leftx.any() and lefty.any() and rightx.any() and righty.any()):
        raise EmptyDataError("Empty data")

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    window_img = np.zeros_like(out_img)
    window_img = draw_window(window_img, left_fitx, ploty, margin)
    window_img = draw_window(window_img, right_fitx, ploty, margin)

    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    return {
        "left_fit": left_fit, "right_fit": right_fit,
        "left_fitx": left_fitx, "right_fitx": right_fitx,
        "ploty": ploty,
        "out_img": result,
        "window_img": window_img,
        "nonzerox": nonzerox, "nonzeroy": nonzeroy,
        "left_lane_inds": left_lane_inds, "right_lane_inds": right_lane_inds,
        "leftx": leftx, "lefty": lefty, "rightx": rightx, "righty": righty,
    }


def draw_window(img, fit_x, y, margin, color=(0, 255, 0)):
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    draw_region(img, fit_x-margin, fit_x+margin, y, color)
    return img


def draw_curve(img, fit_x, y, color=(0, 255, 0), thickness=2):
    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    line_pts = np.array([np.transpose(np.vstack([fit_x, y]))])

    # Draw the lane onto the warped blank image
    cv2.polylines(img, np.int_([line_pts]), False, color, thickness)
    return img


def draw_region(img, left_x, right_x, y, color=(0, 255, 0)):
    left_line = np.array([np.transpose(np.vstack([left_x, y]))])
    right_line = np.array([np.flipud(np.transpose(np.vstack([right_x, y])))])
    line_pts = np.hstack((left_line, right_line))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(img, np.int_([line_pts]), color)
    return img


def measure_curvature(y_eval, x, y, xm_per_pix=1, ym_per_pix=1):
    """Measure lane curvature

    Args:
    y_eval -- y value at which the curvature is evaluated
    y -- fitted y
    x -- fitted x
    xm_per_pix -- meter per pixel of x axis
    ym_per_pix -- meter per pixel of y axis
    """

    # Fit new polynomials to x,y in world space
    fit_cr = np.polyfit(y*ym_per_pix, x*xm_per_pix, 2)
    # Calculate the new radii of curvature
    curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])

    return curverad


def weighted_img(img, initial_img, alpha=0.8, beta=1., lambd=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, lambd)


def apply_clahe(img_gray, limit=2, grid_size=4):
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(grid_size, grid_size))
    res = clahe.apply(img_gray)
    return res
