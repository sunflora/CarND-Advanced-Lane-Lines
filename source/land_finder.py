import numpy as np
import cv2
import matplotlib.pyplot as plt
from image import FSImage
from lane import Lane


class LaneFinder:
    camera = None
    prev_left_lane_fits = []
    prev_right_lane_fits = []
    DEBUG = False

    def __init__(self, camera):
        self.camera = camera
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        # average x values of the fitted line over the last n iterations
        self.bestx = None
        # polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        # polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        # radius of curvature of the line in some units
        self.radius_of_curvature = None
        # distance in meters of vehicle center from the line
        self.line_base_pos = None
        # difference in fit coefficients between last and new fits
        self.diffs = np.array([0, 0, 0], dtype='float')
        # x values for detected line pixels
        self.allx = None
        # y values for detected line pixels
        self.ally = None
        return

    def process_image(self, image):

        #images sent in via VideoFileClip are BGR!!!
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_undistorted = self.camera.undistort_image(image)

        fsImgUndistorted = FSImage(None)
        fsImgUndistorted.set_image(img_undistorted)
        if self.DEBUG:
            plt.imshow(fsImgUndistorted.get_image_object())
            plt.show(block=True)

        undistortedHSV = fsImgUndistorted.to_hsv(None)
        undistortedGray = fsImgUndistorted.to_grey_scale(None)
        white_and_yellow_region = fsImgUndistorted.get_white_and_yellow_region(undistortedHSV)
        sobel_filter_binary= fsImgUndistorted.apply_sobel_x_and_y(undistortedGray, 50, 255)
        image_processed_binary = fsImgUndistorted.process_image(undistortedHSV, undistortedGray, 50, 225)

        roi_boundaries = np.array([[(550, 450), (750, 450), (1150, 700), (150, 700)]], np.int32)
        lane = Lane(roi_boundaries)
        roi = lane.region_of_interest(image_processed_binary, roi_boundaries)

        if self.DEBUG:
            plt.imshow(roi)
            plt.show(block=True)

        src = np.float32(((580.7, 462.2), (703.7, 462.2), (1048.4, 685.8), (252.3, 685.8)))
        des = np.float32(((252.3, 0), (1048.4, 0), (1048.4, 720), (252.3, 720)))

        lane.set_perspective_values(src, des)
        warped = lane.perspective_transform(roi, (roi.shape[1], roi.shape[0]))
        if self.DEBUG:
            plt.imshow(warped)
            plt.show(block=True)

        histogram = np.sum(warped[int(warped.shape[0] / 2):, :], axis=0)

        # convert 1D warped to 3D for colored rectangle drawing, top_down => colored warped
        top_down = np.dstack((warped, warped, warped)) * 255
        top_down = np.array(top_down, dtype=np.uint8)
        if self.DEBUG:
            plt.imshow(top_down)
            plt.show(block=True)

        left_lane_center, right_lane_center, left_lane_fit, right_lane_fit = lane.predict_lanes(top_down, histogram)

        if left_lane_fit is not None:
            self.prev_left_lane_fits.append(left_lane_fit)
        else:
            left_lane_fit = self.prev_left_lane_fits[-1]

        if right_lane_fit is not None:
            self.prev_right_lane_fits.append(right_lane_fit)
        else:
            right_lane_fit = self.prev_left_lane_fits[-1]


        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        ploty = np.linspace(0, top_down.shape[0] - 1, top_down.shape[0])
        left_fitx = left_lane_fit[0] * ploty ** 2 + left_lane_fit[1] * ploty + left_lane_fit[2]
        right_fitx = right_lane_fit[0] * ploty ** 2 + right_lane_fit[1] * ploty + right_lane_fit[2]

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective matrix (Minv)
        newwarp = lane.inverse_perspective_transform(color_warp, (roi.shape[1], roi.shape[0]))

        curve_left, curve_right = lane.get_curvatures(ploty, left_fitx, right_fitx)
        curve_left_text = "Radius of Curve (L): " + np.str(np.around(curve_left, decimals=2)) + " m"
        curve_right_text = "Radius of Curve (R): " + np.str(np.around(curve_right, decimals=2)) + " m"

        image_bottom = image.shape[0]
        left_lane_bottom = left_lane_fit[0] * image_bottom ** 2 + left_lane_fit[1] * image_bottom + left_lane_fit[2]
        right_lane_bottom = right_lane_fit[0] * image_bottom ** 2 + right_lane_fit[1] * image_bottom + right_lane_fit[2]

        from_center = lane.get_deviation_from_center(image.shape[1]/2, left_lane_bottom, right_lane_bottom)
        if from_center > 0:
            from_center_text = "Vehicle is : " + np.str(from_center) + "m right of center"
        else:
            from_center_text = "Vehicle is : " + np.str(from_center*-1) + "m left of center"

        # Combine the result with the original image
        img_undistored_rgb = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2RGB)
        result = cv2.addWeighted(img_undistored_rgb, 1, newwarp, 0.3, 0)


        cv2.putText(result, curve_left_text, (100, 300), color=(255, 255, 255), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, thickness=1)
        cv2.putText(result, curve_right_text, (100, 330), color=(255, 255, 255), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, thickness=1)
        cv2.putText(result, from_center_text, (100, 360), color=(255, 255, 255), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1, thickness=1)

        result[100:190, 100:260] = cv2.resize(img_undistorted, (160, 90))

        white_and_yellow_region_color = np.dstack((white_and_yellow_region, white_and_yellow_region, white_and_yellow_region)) * 255
        result[100:190, 300:460] = cv2.resize(white_and_yellow_region_color, (160, 90))

        sobel_filter_binary_color = np.dstack((sobel_filter_binary, sobel_filter_binary, sobel_filter_binary)) * 255
        result[100:190, 500:660] = cv2.resize(sobel_filter_binary_color, (160, 90))

        processed_binary = np.zeros_like(image_processed_binary).astype(np.uint8)
        image_processed_color = np.dstack((processed_binary, processed_binary, processed_binary))
        image_processed_color[:,:,0] = image_processed_binary * 255
        image_processed_color[:,:,1] = image_processed_binary * 255
        image_processed_color[:,:,2] = image_processed_binary * 255
        result[100:190, 700:860] = cv2.resize(image_processed_color, (160, 90), interpolation=cv2.INTER_AREA)

        warped = np.dstack((warped, warped, warped))
        warped = np.array(warped, dtype=np.uint8) * 255
        result[100:190, 900:1060] = cv2.resize(warped, (160, 90))

        result[100:190, 1100:1260] = cv2.resize(color_warp, (160, 90))

        if self.DEBUG:
            plt.imshow(result)
            plt.show(block=True)

        result = np.array(result, dtype=np.uint32)

        return result
