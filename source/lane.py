import matplotlib.pyplot as plt
import numpy as np
import cv2


class Lane:
    roi_boundaries = None
    perspective_src = None
    perspective_des = None
    perspective_M = None
    perspective_Minv = None

    def __init__(self, roi_boundaries):
        self.roi_boundaries = roi_boundaries

    def draw_roi_vertices_on_image(self, img, filename):
        fig = plt.figure()
        plt.imshow(img)
        for v in self.roi_boundaries:
            plt.plot(v[0], v[1], marker='o', color='r', ls='')
        print("A. filename is ", filename)
        if filename is not None:
            print("B. filename is ", filename)
            fig.subplots_adjust(bottom=0)
            fig.subplots_adjust(top=1)
            fig.subplots_adjust(right=1)
            fig.subplots_adjust(left=0)
            plt.savefig(filename, bbox_inches='tight', pad_inches=0)
            plt.clf()
        else:
            plt.show(block=True)

    def region_of_interest(self, img, vertices):
        """
        Applies an image mask.

        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """

        if vertices is not None:
            self.roi_boundaries = vertices

        # defining a blank mask to start with
        mask = np.zeros_like(img)

        # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, self.roi_boundaries, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def set_perspective_values(self, src, des):
        self.perspective_src = src
        self.perspective_des = des
        self.perspective_M = cv2.getPerspectiveTransform(self.perspective_src, self.perspective_des)
        self.perspective_Minv = cv2.getPerspectiveTransform(self.perspective_des, self.perspective_src)

    def draw_perspective_source_on_image(self, image, filename):

        src = self.perspective_src

        fig = plt.figure()
        plt.imshow(image, cmap='gray')
        plt.plot(src[0][0], src[0][1], marker='o', color='r', ls='')
        plt.plot(src[1][0], src[1][1], marker='o', color='r', ls='')
        plt.plot(src[2][0], src[2][1], marker='o', color='r', ls='')
        plt.plot(src[3][0], src[3][1], marker='o', color='r', ls='')
        fig.subplots_adjust(bottom=0)
        fig.subplots_adjust(top=1)
        fig.subplots_adjust(right=1)
        fig.subplots_adjust(left=0)
        plt.savefig(filename)
        plt.close()

    def perspective_transform(self, image, image_shape):
        warped = cv2.warpPerspective(image, self.perspective_M, image_shape, flags=cv2.INTER_LINEAR)
        warped[warped > 0] = 1
        return warped

    def inverse_perspective_transform(self, image, image_shape):
        warped = cv2.warpPerspective(image, self.perspective_Minv, image_shape, flags=cv2.INTER_LINEAR)
#        warped[warped > 0] = 1
        return warped

    def get_lane_centers(self, histogram):
        midpoint = np.int(histogram.shape[0] / 2)
        left = np.argmax(histogram[:midpoint])
        right = np.argmax(histogram[midpoint:]) + midpoint
        return left, right

    def get_lane(self, image, lane_center, nwindows, margin=100, minpix=50):
        window_height = np.int(image.shape[0] / nwindows)
        nonzero = image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        current = lane_center
        lane_inds = []

        for window in range(nwindows):
            win_y_low = image.shape[0] - (window + 1) * window_height
            win_y_high = image.shape[0] - window * window_height
            win_x_low = current - margin
            win_x_high = current + margin
            cv2.rectangle(image, (win_x_low, win_y_low), (win_x_high, win_y_high), color=(0, 255, 0), thickness=2)
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low)
                         & (nonzerox < win_x_high)).nonzero()[0]
            lane_inds.append(good_inds)
            if len(good_inds) > minpix:
                current = np.int(np.mean(nonzerox[good_inds]))

        lane_inds = np.concatenate(lane_inds)
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]
        try:
            lane_fit = np.polyfit(y, x, 2)
        except:
            lane_fit = None
        return lane_fit

    def get_lanes(self, image, right_center, left_center, nwindows, margin=100, minpix=50 ):
        left_lane = self.get_lane(image, left_center, nwindows, margin=100, minpix=50)
        right_lane = self.get_lane(image, right_center, nwindows, margin=100, minpix=50)
        return left_lane, right_lane

    def predict_lanes(self, image, histogram):
        leftx_base, rightx_base = self.get_lane_centers(histogram)

        lane_left, lane_right = self.get_lanes(image=image, right_center=rightx_base, left_center=leftx_base,
                                               nwindows=9, margin=100, minpix=50)
        return leftx_base, rightx_base, lane_left, lane_right

    def get_curvatures(self, ploty, left_fitx, right_fitx):
        leftx = left_fitx
        rightx = right_fitx
        y_eval = np.max(ploty)

        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30 / 720  # meters per pixel in y dimension
        xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
        # Calculate the new radii of curvature
        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
            2 * right_fit_cr[0])
        # Now our radius of curvature is in meters

        return left_curverad, right_curverad

    def get_deviation_from_center(self, image_center, left_lane_center, right_lane_center):
        xm_per_pix = 3.7 / 700
        from_center_pixel = image_center - (left_lane_center + right_lane_center)/2
        from_center_meter = from_center_pixel * xm_per_pix
        return np.round(from_center_meter, 2)
