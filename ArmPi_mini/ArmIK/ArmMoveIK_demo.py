#!/usr/bin/env python3
# encoding:utf-8
import sys
import time
sys.path.append('/home/pi/ArmPi_mini/')
from ArmIK.ArmMoveIK import *

if sys.version_info.major == 2:
    print('Please run this program with python3!')
    sys.exit(0)
    
print('''
**********************************************************
****************功能:逆运动学上下移动例程********************
**********************************************************
----------------------------------------------------------
Official website:http://www.hiwonder.com
Online mall:https://huaner.tmall.com/
----------------------------------------------------------
Tips:
 * 按下Ctrl+C可关闭此次程序运行，若失败请多次尝试！
----------------------------------------------------------
''')

# 实例化逆运动学库
AK = ArmIK()
 
if __name__ == "__main__":
    '''
    AK.setPitchRangeMoving(coordinate_data, alpha, alpha1, alpha2, movetime):
    给定坐标coordinate_data和俯仰角alpha,以及俯仰角范围的范围alpha1, alpha2，自动寻找最接近给定俯仰角的解，并转到目标位置
    如果无解返回False,否则返回舵机角度、俯仰角、运行时间
    坐标单位cm， 以元组形式传入，例如(0, 5, 10)
    alpha: 为给定俯仰角
    alpha1和alpha2: 为俯仰角的取值范围
    movetime:为舵机转动时间，单位ms, 如果不给出时间，则自动计算    
    '''
    AK.setPitchRangeMoving((0, 6, 18), 0,-90, 90, 1500) # 设置机械臂初始位置(x:0, y:6, z:18),运行时间:1500毫秒
    time.sleep(1.5) # 延时1.5秒
    
    for i in range(2): # for循环运行2次
        AK.setPitchRangeMoving((0, 6, 22), 0,-90, 90, 1000) # 设置机械臂上移到位置(x:0, y:6, z:22),运行时间:1000毫秒
        time.sleep(1.2) # 延时1.2秒
        
        AK.setPitchRangeMoving((0, 6, 18), 0,-90, 90, 1000) # 设置机械臂下移到初始位置,运行时间:1000毫秒
        time.sleep(1.2) # 延时1.2秒
    