import cv2
import os
import numpy as np


class FSImage:
    full_filename = None
    image = None
    image_shape = None

    def __init__(self, full_filename):
        if full_filename is not None:
            self.full_filename = full_filename
            self.image = cv2.imread(full_filename)
            self.image_shape = (self.image.shape[1], self.image.shape[0])

    def set_image(self,image):
        self.full_filename = None
        self.image = image
        self.image_shape = (self.image.shape[1], self.image.shape[0])

    def get_image_object(self):
        return self.image

    def get_image_shape(self):
        if self.image_shape is None:
            self.image_shape = (self.image.shape[1], self.image.shape[0])
        return self.image_shape

    def get_image_filename_base(self):
        return os.path.basename(self.full_filename)

    def save_image(self, output_filename):
        cv2.imwrite(output_filename,self.image)

    def to_grey_scale(self, img, original_format="BGR"):
        if img is not None:
            self.set_image(img)
        new_image = None
        if original_format == "BGR":
            new_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        elif original_format == "RGB":
            new_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        else:
            print("ALERT: In FSImage.to_grey_scale, unknown original format, please re-specify!")
        return new_image

    def to_hsv(self, img, original_format="BGR"):
        if img is not None:
            self.set_image(img)
        new_image = None
        if original_format == "BGR":
            new_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        elif original_format == "RGB":
            new_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV)
        else:
            print("ALERT: In FSImage.to_hsv, unknown original format, please re-specify!")
        return new_image

    def to_rgb(self, img, original_format="BGR"):
        if img is not None:
            self.set_image(img)
        new_image = None
        if original_format == "BGR":
            new_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        else:
            print("ALERT: In FSImage.to_rgb, unknown original format, please re-specify!")
        return new_image

    def get_yellow_region_binary(self, hsv):

        # lower_yellow = np.array([0, 100, 100])
        lower_yellow = np.array([0, 100, 200])
        # upper_yellow = np.array([50, 255, 255])
        upper_yellow = np.array([40, 255, 255])

        yellow_region = cv2.inRange(hsv, lower_yellow, upper_yellow)
        binary = np.zeros(hsv.shape[0:2])
        binary[(yellow_region > 0)] = 1
        return binary

    def get_white_region_binary(self, hsv):
        lower_white = np.array([20, 0, 180])
        upper_white = np.array([255, 80, 255])
        white_region = cv2.inRange(hsv, lower_white, upper_white)
        binary = np.zeros(hsv.shape[0:2])
        binary[(white_region > 0)] = 1
        return binary

    def get_white_region_binary_rgb(self, rgb):
        lower_white = np.array([120, 120, 180])
        upper_white = np.array([255, 255, 255])
        white_region = cv2.inRange(rgb, lower_white, upper_white)
        binary = np.zeros(rgb.shape[0:2])
        binary[(white_region > 0)] = 1
        return binary

    def apply_absolute_sobel(self, gray, orient, thresh_min=0, thresh_max=255):
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        abs_sobel = np.absolute(sobel)
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        binary_output = (scaled_sobel > thresh_min) & (scaled_sobel < thresh_max)
        return binary_output

    def apply_sobel_x_and_y(self, gray, thresh_min, thresh_max):
        binary_x = self.apply_absolute_sobel(gray, 'x', thresh_min, thresh_max)
        binary_y = self.apply_absolute_sobel(gray, 'y', thresh_min, thresh_max)
        binary = np.zeros_like(gray)
        binary[((binary_x == 1) & (binary_y == 1))] = 1
        return binary

    def process_image(self, hsv, gray, sobel_min, sobel_max):
        binary_white_or_yellow = self.get_white_and_yellow_region(hsv)
        binary_sober_x_and_y = self.apply_sobel_x_and_y(gray, thresh_min=sobel_min, thresh_max=sobel_max)
        binary_process_image = np.zeros_like(binary_white_or_yellow)
        binary_process_image[(binary_white_or_yellow==1) | (binary_sober_x_and_y == 1)] = 1
        return binary_process_image

    def get_white_and_yellow_region_old(self, hsv):
        binary_white = self.get_white_region_binary(hsv)
        binary_yellow = self.get_yellow_region_binary(hsv)
        binary = np.zeros(hsv.shape[0:2])
        binary[(binary_white == 1) | (binary_yellow == 1)] = 1
        return binary

    def get_white_and_yellow_region(self, hsv):
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        binary_white = self.get_white_region_binary_rgb(rgb)
        binary_yellow = self.get_yellow_region_binary(hsv)
        binary = np.zeros(hsv.shape[0:2])
        binary[(binary_white == 1) | (binary_yellow == 1)] = 1
        return binary



