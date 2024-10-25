#!/usr/bin/python3
# coding=utf8
import time
import Board
import ActionGroupControl as AGC

print('''
*Action Group Sequence
''')

AGC.runAction('armAG_base')
AGC.runAction('armAG_pickup')
AGC.runAction('armAG_placeBlue')
AGC.runAction('armAG_pickup')
AGC.runAction('armAG_placeGreen')
AGC.runAction('armAG_pickup')
AGC.runAction('armAG_placeRed')
AGC.runAction('armAG_base')
