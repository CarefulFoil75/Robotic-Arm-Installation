#!/usr/bin/env python3
# encoding:utf-8

"""
File name:      pi_camera.py
Author:         Michael Johnson
Date:           8/23/2024
Description:    ArmPi camera setup and methods for viewing live video and capturing images.
"""

# Get current working directory.
import os
from pathlib import Path
curr_dir = Path(os.getcwd())
# curr_dir /= 'ArmPi_windowsMJ'

# Append current working directory to file path so files in directory can easily be referenced.
import sys
sys.path.append(str(curr_dir))

# Import library packages.
import cv2
import time
# import threading
import numpy as np

# Import content from files.
# from ArmPi_windowsMJ.CalibrationConfig_win import *

# Check Python version.
if sys.version_info.major == 2:
    print('Please run this program with Python3!')
    sys.exit(0)

# ----- File variables -----
camera_calib_path = curr_dir / 'camera_calibration'
camera_calib_file = camera_calib_path / 'calibration_param.npz'

save_path = curr_dir / 'data'
if not os.path.exists(save_path):
    os.mkdir(save_path)

train_path = save_path / 'training_images'
test_path = save_path / 'testing_images'

if not os.path.exists(train_path):
    os.mkdir(train_path)
if not os.path.exists(test_path):
    os.mkdir(test_path)



# ----- Other global variables -----
window_name = 'PiCam Vision'

# set debug to False to not display screens
debug = True
# set capture to False to disable taking pictures
capture = True


class PiCamera:
    def __init__(self, camera_ind=0, resolution=(640, 480)):
        self.cap = None
        self.width = resolution[0]
        self.height = resolution[1]
        self.frame = None
        self.opened = False
        self.camera_ind = camera_ind

        # Loading and parameters
        self.param_data = np.load(str(camera_calib_file))
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
            self.cap = cv2.VideoCapture(self.camera_ind)
            # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_SATURATION, 40)
            self.opened = True
            _, thor = self.cap.read()
            print(thor.shape)
        except Exception as e:
            print('Failed to open the camera:', e)

    def camera_close(self):
        try:
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
                    cap = cv2.VideoCapture(self.camera_ind)
                    ret, _ = cap.read()
                    if ret:
                        self.cap = cap
            elif self.opened:
                cap = cv2.VideoCapture(self.camera_ind)
                ret, _ = cap.read()
                if ret:
                    self.cap = cap
            else:
                time.sleep(0.01)
        except Exception as e:
            print('Error in getting camera image:', e)
            time.sleep(0.01)

    def mouse_event(self, event, x, y, flags, param):
        # The mouse event is connected to the window pop-up and triggers the event when the user clicks on the image.
        if event == cv2.EVENT_LBUTTONDOWN:
            print("LEFT CLICKED:")
            print(x, y)

if __name__ == '__main__':
    my_camera = PiCamera(camera_ind=1)
    my_camera.camera_open()
    # i = 0
    # while i < 10:
    while True:
        my_camera.camera_task()
        img = my_camera.frame
        # i += 1
        if img is not None:
            cv2.imshow(window_name, img)
            key = cv2.waitKey(1)
            if key == 27:
                break
    my_camera.camera_close()
    cv2.destroyAllWindows()
