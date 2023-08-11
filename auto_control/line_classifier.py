import numpy as np
import cv2 as cv
from time import time


class LineClassifier:
    def __init__(self):
        self.box_width = 30
        self.box_height = 15
        self.prob_max_side = 0.8
        self.prob_min_mid = 0.9

    def MakeBoxAroundLine(self, line):
        delta_x = line[2] - line[0]
        delta_y = line[3] - line[1]
        upperL_corner = (line[0] - int((self.box_width - delta_x) / 2), line[1] -
                         int((self.box_height - delta_y) / 2))
        lowerR_corner = (line[2] + int((self.box_width - delta_x) / 2), line[3] +
                         int((self.box_height - delta_y) / 2))
        return upperL_corner, lowerR_corner

    def fitLine(self, pt1, pt2, epsilon=1):
        x1, y1 = pt1
        x2, y2 = pt2
        if x1 != x2:
            slope = (y1 - y2) / (x1 - x2)
            intercept = y1 - slope * x1
        else:
            slope = (y1 - y2 + epsilon) / (x1 - x2 + epsilon)
            intercept = y1 - slope * x1
        # slope = 0.000001 if slope < 0.000001 else slope
        return slope, intercept

    def ClassifyLine(self, lines, runable_lane):
        mid_lines = []
        left_lines = []
        right_lines = []
        mid_lines_coordinate = []

        try:
            print(len(lines))
            for line in lines:
                start = time()
                x1, y1, x2, y2 = line.reshape(4)
                slope, intecept = self.fitLine((x1, y1), (x2, y2))
                upperL_corner, lowerR_corner = self.MakeBoxAroundLine([x1, y1, x2, y2])
                crop_road = runable_lane[upperL_corner[1]:lowerR_corner[1], upperL_corner[0]:lowerR_corner[0]]
                # print("crop",crop_road)
                num_of_pixel_road = cv.countNonZero(crop_road)
                prob_road = num_of_pixel_road / (self.box_height * self.box_width)
                end = time()
                # print("FPS count: ", 1/(end-start))
                start = time()
                if prob_road < self.prob_max_side:
                    if slope < -0.1:
                        left_lines.append((slope, intecept))
                    elif slope > 0.1:
                        right_lines.append((slope, intecept))
                elif prob_road > self.prob_min_mid:
                    mid_lines.append((slope, intecept))
                    mid_lines_coordinate.append((x1, y1, x2, y2))
                end = time()
                # print("FPS fit: ", 1/(end-start))
        except TypeError:
            pass
        return left_lines, mid_lines, right_lines, mid_lines_coordinate