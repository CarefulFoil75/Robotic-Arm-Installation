#!/usr/bin/env python3
# encoding:utf-8

"""
File name:      take_dev_images.py
Author:         Michael Johnson
Date:           9/6/2024
Description:    Developer script for taking training images. FOR DEVELOPER USE ONLY.
"""

# Import library packages.
import os
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

    initials = input("Enter your initials: ")
    color = input("Enter the color of the block you are taking images of: ")
    updated_img_ctr = input("Enter the camera counter index if you want to update it. If not, press 'Enter' to continue: ")

    if updated_img_ctr == '' or int(updated_img_ctr) < 1:
        updated_img_ctr = 1

    init_path = cc.dev_path / initials
    color_path = init_path / color

    if not os.path.exists(init_path):
        os.mkdir(init_path)
    if not os.path.exists(color_path):
        os.mkdir(color_path)

    # Create camera object
    my_camera = PiCamera(camera_index=1)
    my_camera.saved_img_counter = int(updated_img_ctr)

    # Create image window and set mouse callback
    cv2.namedWindow(cc.window_name)
    cv2.setMouseCallback(cc.window_name, my_camera.mouse_event, [color_path, color, initials])

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
            if key == ord('c'):
                my_camera.take_image(color_path, color, initials)
            elif key == 27:
                break
    my_camera.camera_close()
    cv2.destroyAllWindows()
