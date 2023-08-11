import numpy as np
from time import time


class ProjectData:
    car_rotation = 0

    currentMode = 0
    car_rotation_predict = 0
    total = 0
    count = 0
    numLine = 0
    cam_rotation = 0
    car_speed = 60
    goodDistance = 84

    limitTimeMode2 = 5
    startTime = 0
    startMode2 = False

    hasSign = False
    leftSign = False
    rightSign = False
    isNearCross = False
    isFindingRoad = False

    startRotateLeft = False
    startRotateRight = False

    lookingRoad = False
    twoLine = False

    frame = 0

    haveLineL = False
    haveLineR = False

    rotateInBigAngle = False
    area = 0
    isStart = True
    timeStartGame = time()
    isSpeedUp = False
    countDown = 3
    hasSignal = False
    reasonNoSignal = 1

    obstacle_left = 0
    obstacle_right = 0

    box_increase_road = 10
    box_increase_side = 30
    box_area_thresh = 1000
    box_increase_below = 20

    modify_center = float(1) / float(3)
    thresh_modify = 400

    rgb_image = np.zeros((240, 320, 3), np.uint8)