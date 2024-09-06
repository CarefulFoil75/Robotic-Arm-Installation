#!/usr/bin/env python3
# encoding:utf-8

"""
File name:      pi_camera.py
Author:         Michael Johnson
Date:           8/23/2024
Description:    ArmPi camera setup and methods for viewing live video and capturing images.
"""

# Import library packages.
import sys
import cv2
import time
import numpy as np
# import threading

import common_code_windows_MJ as cc

# Append current working directory to file path so files in directory can easily be referenced.
sys.path.append(str(cc.curr_dir))

# Check Python version.
if sys.version_info.major == 2:
    print('Please run this program with Python3!')
    sys.exit(0)


class PiCamera:
    def __init__(self, camera_index=0, resolution=(640, 480)):
        self.cap = None
        self.width = resolution[0]
        self.height = resolution[1]
        self.frame = None
        self.opened = False
        self.camera_index = camera_index

        # Image saving variables
        self.saved_img_counter = 1
        self.captured_imgs = [] # Could make this a numpy array in the future, but lists are easier for quick manipulation

        # Load calibration parameters and undistort camera
        self.param_data = np.load(str(cc.camera_calib_file))
        self.dim = tuple(self.param_data['dim_array'])
        self.k = np.array(self.param_data['k_array'].tolist())
        self.d = np.array(self.param_data['d_array'].tolist())
        self.p = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(self.k, self.d, self.dim ,None)
        self.Knew = self.p.copy()
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(self.k, self.d, np.eye(3), self.Knew, self.dim, cv2.CV_16SC2)
        
        # self.th = threading.Thread(target=self.camera_task, args=(), daemon=True)
        # self.th.start()

    def camera_open(self):
        try:
            print('opening camera!')
            self.cap = cv2.VideoCapture(self.camera_index)
            _, debug_test = self.cap.read() # For some reason, self.cap loses a dimension of the image while setting its properties if a test read is not performed after initializing the camera.
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('Y', 'U', 'Y', 'V'))
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_SATURATION, 40)
            # print(debug_test.shape)
            self.opened = True

        except Exception as e:
            print('Failed to open the camera:', e)

    def camera_close(self):
        try:
            print(f'{len(self.captured_imgs)} image(s) saved.')
            print('closing camera!')
            self.opened = False
            time.sleep(0.2)
            if self.cap is not None:
                self.cap.release()
                time.sleep(0.05)
            self.cap = None
        except Exception as e:
            print('Failed to close the camera:', e)

    def camera_task(self):
        # print('camera task started!')
        try:
            if self.opened and self.cap.isOpened():
                ret, raw_img = self.cap.read()
                if ret:
                    correct_img = cv2.remap(raw_img, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                    self.frame = cv2.resize(correct_img, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                else:
                    self.frame = None
                    cap = cv2.VideoCapture(self.camera_index)
                    ret, _ = cap.read()
                    if ret:
                        self.cap = cap
            elif self.opened:
                cap = cv2.VideoCapture(self.camera_index)
                ret, _ = cap.read()
                if ret:
                    self.cap = cap
            else:
                time.sleep(0.01)
        except Exception as e:
            print('Error in getting camera image:', e)
            time.sleep(0.01)

    def take_image(self):
        cv2.imwrite(str(cc.imgs_path / f'{self.saved_img_counter:04}.jpg'), self.frame)
        self.captured_imgs.append(self.frame)
        self.saved_img_counter += 1

    def mouse_event(self, event, x, y, flags, param):
        # The mouse event is connected to the window pop-up and triggers the event when the user clicks on the image.
        if event == cv2.EVENT_LBUTTONDOWN:
            self.take_image()
            print("Image Captured")


if __name__ == '__main__':
    # Create directories from common code file save paths
    cc.make_directories()

    # Create camera object
    my_camera = PiCamera(camera_index=1)

    # Create image window and set mouse callback
    cv2.namedWindow(cc.window_name)
    cv2.setMouseCallback(cc.window_name, my_camera.mouse_event)

    # Open camera
    my_camera.camera_open()
    # i = 0
    # while i < 10:
    while True:
        my_camera.camera_task()
        img = my_camera.frame
        # i += 1
        if img is not None:
            cv2.imshow(cc.window_name, img)
            key = cv2.waitKey(1)
            if key == 27:
                break
    my_camera.camera_close()
    cv2.destroyAllWindows()
