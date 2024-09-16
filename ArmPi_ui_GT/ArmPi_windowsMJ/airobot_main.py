#!/usr/bin/env python3
# encoding:utf-8

"""
File name:      airobot_main.py
Author:         Michael Johnson
Date:           9/6/2024
Description:    Main file for AI Robot system operation.
"""

# Import library packages.
import sys
import cv2
import time
import numpy as np
# import threading

import common_code_windows_MJ as cc
from pi_camera import PiCamera

# Append current working directory to file path so files in directory can easily be referenced.
sys.path.append(str(cc.curr_dir))

# Check Python version.
if sys.version_info.major == 2:
    print('Please run this program with Python3!')
    sys.exit(0)

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

    while True:
        my_camera.camera_task()
        img = my_camera.frame
        if img is not None:
            cv2.imshow(cc.window_name, img)
            key = cv2.waitKey(1)
            if key == 27: # ASCII code for 'Esc' key
                break
    my_camera.camera_close()
    cv2.destroyAllWindows()