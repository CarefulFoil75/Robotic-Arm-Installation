#!/usr/bin/env python3
# encoding:utf-8

"""
File name:      take_images.py
Author:         Brattin Miller Patton
Date:           9/11/2024
Description:    UI backend camera function
"""

# Import library packages.
import os
import sys
import cv2
import time
import importlib
import numpy as np
# import threading

import common_code_windows_MJ as cc
sys.path.append(str(cc.curr_dir))
from pi_camera import PiCamera

# Append current working directory to file path so files in directory can easily be referenced.
#sys.path.append(str(cc.curr_dir))

# Check Python version.
if sys.version_info.major == 2:
    print('Please run this program with Python3!')
    sys.exit(0)

def take_image(folder_name, color, name = None):
    init_path = cc.imgs_path / folder_name
    color_path = init_path / color

    if not os.path.exists(init_path):
        os.mkdir(init_path)
    if not os.path.exists(color_path):
        os.mkdir(color_path)

    my_camera = PiCamera(camera_index=1)
    my_camera.camera_open()
    my_camera.mouse_event
    my_camera.camera_close()

#take_image("ed", "blue", "0_blue")