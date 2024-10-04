#!/usr/bin/env python3
# encoding:utf-8

"""
File name:      common_code_windows_MJ.py
Author:         Michael Johnson
Date:           9/6/2024
Description:    Common variables and functions used across AI Robot Arm system files.
"""

# Get current working directory.
import os
import sys
from pathlib import Path

curr_dir = Path(os.path.abspath(os.path.dirname(__file__))) #abspath() returns the directory of the file
#curr_dir = Path(os.getcwd()) #getcwd() returns the directory of the executor
# curr_dir /= 'ArmPi_windowsMJ'

# ----- File variables -----
camera_calib_path = curr_dir / 'camera_calibration'
camera_calib_file = camera_calib_path / 'calibration_param.npz'

save_path = curr_dir / 'data'

dev_path = save_path / 'dev_images'

imgs_path = save_path / 'all_images'
train_path = save_path / 'training_images'
test_path = save_path / 'testing_images'

# ----- Other global variables -----
window_name = 'PiCam Vision'

# set debug to False to not display screens
debug = True
# set capture to False to disable taking pictures
capture = True

def make_directories():
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(dev_path):
        os.mkdir(dev_path)
    if not os.path.exists(imgs_path):
        os.mkdir(imgs_path)
    if not os.path.exists(train_path):
        os.mkdir(train_path)
    if not os.path.exists(test_path):
        os.mkdir(test_path)