import numpy as np
import cv2
import matplotlib.pyplot as plt
from camera import Camera
from image import FSImage
from lane import Lane


def save_plot(img, cmap, filename):
    fig = plt.figure()
    fig.subplots_adjust(bottom=0)
    fig.subplots_adjust(top=1)
    fig.subplots_adjust(right=1)
    fig.subplots_adjust(left=0)
    if cmap == 'gray':
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img)
    plt.savefig(filename)
    plt.close()


def plot_predicted_lanes(top_down, left_lane_fit, right_lane_fit, filename):
    # Generate x and y values for plotting
    ploty = np.linspace(0, top_down.shape[0] - 1, top_down.shape[0])
    left_fitx = left_lane_fit[0] * ploty ** 2 + left_lane_fit[1] * ploty + left_lane_fit[2]
    right_fitx = right_lane_fit[0] * ploty ** 2 + right_lane_fit[1] * ploty + right_lane_fit[2]

    plt.imshow(top_down)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    plt.savefig(filename)
    #plt.show(block=True)
    plt.close()

test_images = ['straight_lines1','straight_lines2','test1','test2','test3','test4','test5','test6']

# Camera calibration
camera = Camera((1280, 720), 9,6)
camera.calibrate('../camera_cal/*.jpg', '../output_images/camera_cal/draw_corners/')
camera.undistort_images('../camera_cal/*.jpg','../output_images/camera_cal/undistorted/')

for t in test_images:

    test_output_folder = "../output_images/test_images/" + t + "/"

    fsImgOrig = FSImage(full_filename="../test_images/" + t + ".jpg")
    img_orig = fsImgOrig.get_image_object()

    img_undistorted = camera.undistort_image(img_orig)
    fsImgUndistorted = FSImage(None)
    fsImgUndistorted.set_image(img_undistorted)
    fsImgUndistorted.save_image(test_output_folder +  "01-undistorted.jpg")

    undistortedHSV = fsImgUndistorted.to_hsv(None)
    white_region_binary = fsImgUndistorted.get_white_region_binary(undistortedHSV)
    save_plot(white_region_binary, cmap='gray', filename=test_output_folder + "02-white-region.png")

    yellow_region_binary = fsImgUndistorted.get_yellow_region_binary(undistortedHSV)
    save_plot(yellow_region_binary, cmap='gray', filename=test_output_folder + "03-yellow-region.png")

    white_and_yellow_region = fsImgUndistorted.get_white_and_yellow_region(undistortedHSV)
    save_plot(white_and_yellow_region, cmap='gray', filename = test_output_folder + "04-white-and-yellow-region.png")

    undistortedGray = fsImgUndistorted.to_grey_scale(None)
    sobel_binary_x = fsImgUndistorted.apply_absolute_sobel(undistortedGray, orient='x', thresh_min=50, thresh_max=225)
    save_plot(sobel_binary_x, cmap='gray', filename = test_output_folder + "05-sobel-x.png")
    sobel_binary_y = fsImgUndistorted.apply_absolute_sobel(undistortedGray, orient='y', thresh_min=50, thresh_max=225)
    save_plot(sobel_binary_y, cmap='gray', filename = test_output_folder + "06-sobel-y.png")
    sobel_result = np.zeros_like(undistortedGray)
    sobel_result[((sobel_binary_x == 1) & (sobel_binary_y == 1))] = 1
    save_plot(sobel_result, cmap='gray', filename=test_output_folder + "07-sobel-x-and-y.png")

    processed_image_and = np.zeros_like(undistortedGray)
    processed_image_or = np.zeros_like(undistortedGray)
    processed_image_and[(sobel_result == 1) & (white_and_yellow_region == 1)] = 1
    processed_image_or[(sobel_result == 1) | (white_and_yellow_region == 1)] = 1

    #save_plot(processed_image_and, cmap='gray', filename=test_output_folder + "06-processed-image-and.png")
    save_plot(processed_image_or, cmap='gray', filename=test_output_folder + "08-color-sobel-or.png")


    roi_boundaries = np.array([[(550, 450), (750, 450), (1150, 700), (150, 700)]], np.int32)
    lane = Lane(roi_boundaries)
#    roi = lane.region_of_interest(white_and_yellow_region, roi_boundaries)
    roi = lane.region_of_interest(processed_image_or, roi_boundaries)
    save_plot(roi, cmap='gray', filename = test_output_folder + "09-roi.png")

    src = np.float32(((580.7, 462.2), (703.7, 462.2), (1048.4, 685.8), (252.3, 685.8)))
#    des = np.float32(((252.3, 100), (1048.4, 100), (1048.4, 685.8), (252.3, 685.8)))
    des = np.float32(((252.3, 0), (1048.4, 0), (1048.4, 720), (252.3, 720)))

    lane.set_perspective_values(src, des)

    lane.draw_perspective_source_on_image(roi, filename= test_output_folder + "10-roi-perspective-source.png")
    warped = lane.perspective_transform(roi, (roi.shape[1], roi.shape[0]))

    save_plot(warped, cmap='gray', filename=test_output_folder + "11-perspective-transform.png")

    histogram = np.sum(warped[int(warped.shape[0]/2):,:], axis=0)
    plt.plot(histogram)
    plt.savefig(filename= test_output_folder + "12-histogram.png")
    plt.close()

    # convert 1D warped to 3D for colored rectangle drawing, top_down => colored warped
    top_down = np.dstack((warped, warped, warped)) * 255
    top_down = np.array(top_down, dtype=np.uint8)

    left_center, right_center, left_lane_fit, right_lane_fit = lane.predict_lanes(top_down, histogram)

    plot_predicted_lanes(top_down, left_lane_fit, right_lane_fit, filename=test_output_folder + "13-line-fit-projection.png")

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
    newwarp = lane.inverse_perspective_transform(color_warp, (roi.shape[1],roi.shape[0]))

    curve_left, curve_right = lane.get_curvatures(ploty, left_fitx, right_fitx)
    curve_left_text = "Left lane curvature: " + np.str(np.around(curve_left, decimals=2)) + " m."
    curve_right_text = "Right lane curvature: " + np.str(np.around(curve_right, decimals=2)) + " m."

    image_bottom = img_undistorted.shape[1]
    left_lane_bottom = left_lane_fit[0] * image_bottom ** 2 + left_lane_fit[1] * image_bottom + left_lane_fit[2]
    right_lane_bottom = right_lane_fit[0] * image_bottom ** 2 + right_lane_fit[1] * image_bottom + right_lane_fit[2]

    from_center = lane.get_deviation_from_center(img_undistorted.shape[1] / 2, left_lane_bottom, right_lane_bottom)
    #        from_center = self.get_deviation_from_center(image.shape[0] / 2, left_lane_center, right_lane_center)
    if from_center > 0:
        from_center_text = "Vehicle is : " + np.str(from_center) + "m right of center"
    else:
        from_center_text = "Vehicle is : " + np.str(from_center * -1) + "m left of center"

    # Combine the result with the original image
    img_undistored_rgb = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2RGB)
    result = cv2.addWeighted(img_undistored_rgb, 1, newwarp, 0.3, 0)

    cv2.putText(result, curve_left_text, (100, 300), color=(255, 255, 255), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, thickness=1)
    cv2.putText(result, curve_right_text, (100, 330), color=(255, 255, 255), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, thickness=1)
    cv2.putText(result, from_center_text, (100, 360), color=(255, 255, 255), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, thickness=1)

    plt.imshow(result)
    save_plot(result, cmap= None, filename= test_output_folder + "14-detected-road-region.png")
#    plt.show(block=True)




