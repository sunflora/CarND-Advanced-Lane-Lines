"""
Camera calibration: use a set of chessboard images to calibrate for camera distortion.

When a camera takes a picture of a 3D object, the resulting 2D photo might be distorted due to effects of perspectives
on the camera lens. For example, extra curvature might occur for objects appears near the edges of the photo.
This could cause problem for self-driving car when it is using the photo taken by on board cameras to identify
the world around it.  Specifically, a straight lane in the real world could be seen as a curved lane
and we want to un-distort the effects via this step of camera calibration.

We figure out the camera matrix and distortion coefficients by studying the chessboard corners and comparing them
with undistorted corners.  We then use these parameters to un-distort the images taken by the same camera.
"""

import numpy as np
import cv2
import glob
import os

class Camera:
    obj_points = []   # 3D points of actual object
    img_points = []   # 2D points of actual object in an image
    mtx = None
    dist = None
    nx = None
    ny = None
    image_shape = None

#    def __init__(self):
#        print("init")

    def __init__(self, image_shape, nx, ny):
        self.image_shape = image_shape
        self.nx = nx
        self.ny = ny
        self.default_object_points = np.zeros((ny * nx, 3), np.float32)
        self.default_object_points[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

    def calibrate(self, input_image_path_expression, output_draw_corners_folder):
        print("Starting camera calibration for images: ", input_image_path_expression)
        image_filenames = glob.glob(input_image_path_expression)

        count = 0
        # Step through the list and search for chessboard corners
        for idx, fname in enumerate(image_filenames):
            fname_base = os.path.basename(fname)
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (self.nx, self.ny), None)

            # If found, add object points, image points
            if ret:
                count = count + 1
                self.obj_points.append(self.default_object_points)
                self.img_points.append(corners)

                # Draw and save the images with corners
                cv2.drawChessboardCorners(img, (self.nx, self.ny), corners, ret)
                draw_corners_filename = output_draw_corners_folder + fname_base
                cv2.imwrite(draw_corners_filename, img)

        print("-- ", self.nx, "x", self.ny, "corners has been found on ", count, "images.")
        print("--  Images with corners drawn are saved under ", output_draw_corners_folder)

        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_points, self.img_points, self.image_shape, None, None)
        print("Camera calibration has completed.")

    def undistort_images(self, input_image_path_expression, output_undistorted_folder):
        print("Applying distortion correction to images: ", input_image_path_expression)
        image_filenames = glob.glob(input_image_path_expression)

        # Step through the list and undistort images
        for indx, fname in enumerate(image_filenames):
            fname_base = os.path.basename(fname)

            img = cv2.imread(fname)

            dst = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
            output_filename = output_undistorted_folder + fname_base
            cv2.imwrite(output_filename, dst)
        print("Distortion correction is completed and images are saved under: ", output_undistorted_folder)

    def undistort_image(self, img):
        undistorted = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return undistorted

