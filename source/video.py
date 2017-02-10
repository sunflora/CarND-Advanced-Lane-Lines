from camera import Camera
from moviepy.editor import VideoFileClip
from land_finder import LaneFinder

# Camera calibration
camera = Camera((1280, 720), 9, 6)
camera.calibrate('../camera_cal/*.jpg', '../output_images/camera_cal/draw_corners/')
camera.undistort_images('../camera_cal/*.jpg','../output_images/camera_cal/undistorted/')

lf = LaneFinder(camera)

#video = ["project_video.mp4", "challenge_video.mp4", "harder_challenge_video.mp4" ]

video = ["project_video.mp4"]

for v in video:
    clip = VideoFileClip("../" + v, audio=False)
    print("clip.duration: ", clip.duration)
    result = clip.fl_image(lf.process_image)
    result.write_videofile('../output_videos/' + v, audio=False)
