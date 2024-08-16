#!/usr/bin/python3
# coding=utf8
import sys
sys.path.append('/home/pi/ArmPi_mini/')
import time
import HiwonderSDK.Board as Board

if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)
    
print('''
**********************************************************
**********************PWM舵机测试**************************
**********************************************************
----------------------------------------------------------
Official website:https://www.hiwonder.com
Online mall:https://hiwonder.tmall.com
----------------------------------------------------------
Tips:
 * 按下Ctrl+C可关闭此次程序运行，若失败请多次尝试！
----------------------------------------------------------
''')

Board.setPWMServoPulse(1, 1650, 300) 
time.sleep(0.3)
Board.setPWMServoPulse(1, 1500, 300) 
time.sleep(0.3)
Board.setPWMServoPulse(1, 1650, 300) 
time.sleep(0.3)
Board.setPWMServoPulse(1, 1500, 300) 
time.sleep(1.5)

Board.setPWMServoPulse(3, 645, 300) 
time.sleep(0.3)
Board.setPWMServoPulse(3, 745, 300) 
time.sleep(0.3)
Board.setPWMServoPulse(3, 695, 300) 
time.sleep(1.5)

Board.setPWMServoPulse(4, 2365, 300) 
time.sleep(0.3)
Board.setPWMServoPulse(4, 2465, 300) 
time.sleep(0.3)
Board.setPWMServoPulse(4, 2415, 300) 
time.sleep(1.5)

Board.setPWMServoPulse(5, 730, 300) 
time.sleep(0.3)
Board.setPWMServoPulse(5, 830, 300) 
time.sleep(0.3)
Board.setPWMServoPulse(5, 780, 300) 
time.sleep(1.5)

Board.setPWMServoPulse(6, 1450, 300) 
time.sleep(0.3)
Board.setPWMServoPulse(6, 1550, 300) 
time.sleep(0.3)
Board.setPWMServoPulse(6, 1500, 300) 
time.sleep(1.5)
