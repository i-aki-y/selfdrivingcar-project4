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

    if (len(right_lane_inds) == 0) or (len(left_lane_inds) == 0):
        raise EmptyDataError("Empty data")

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

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


def update_fit_lane(binary_warped, fit_result):

    left_fit = fit_result["left_fit"]
    right_fit = fit_result["right_fit"]

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

    if (len(right_lane_inds) == 0) or (len(left_lane_inds) == 0):
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

    # create image of fitted curve
    line_img = np.zeros_like(out_img)
    line_img = draw_curve(line_img, left_fitx, ploty)
    line_img = draw_curve(line_img, right_fitx, ploty)

    fit_region_img = np.zeros_like(out_img)
    fit_region_img = draw_region(fit_region_img, left_fitx, right_fitx, ploty)

    return {
        "left_fit": left_fit, "right_fit": right_fit,
        "ploty": ploty,
        "out_img": result,
        "window_img": window_img,
        "line_img": line_img,
        "fit_region_img": fit_region_img,
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


def measure_curvature(y_eval, fit_result, xm_per_pix=1, ym_per_pix=1):
    """Measure lane curvature

    Args:

    y_eval -- y value at which the curvature is evaluated
    fit_result -- dictionary which have fitting result.
    """
    lefty = fit_result["lefty"]
    leftx = fit_result["leftx"]
    rightx = fit_result["rightx"]
    righty = fit_result["righty"]

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_curverad, right_curverad


def weighted_img(img, initial_img, alpha=0.8, beta=1., lambd=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, lambd)


def get_clahe(limit=20, grid_size=4):
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=(grid_size, grid_size))
    return clahe


def apply_clahe(img, clahe):
    if len(img.shape) == 2:
        img = clahe.apply(img)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        img[:, :, 2] = clahe.apply(img[:, :, 2])
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
    return img
