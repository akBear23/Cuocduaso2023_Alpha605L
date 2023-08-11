# import numpy as np
# import cv2 as cv
# import math
# from std_msgs.msg import Float32
# from project_data import ProjectData as pd
# from time import time
#
# class CarControl():
#     def __init__(self, road):
#         # self.sign = sign
#         self.road = road
#         self.carPos_x = 120
#         self.carPos_y = 300
#
#         # create rgb visualize for center
#         center_background_shape = list(self.road.shape)
#         center_background_shape.append(3)
#         self.center = np.zeros(tuple(center_background_shape), np.uint8)
#         self.center[:, :, 0] = self.road
#
#         self.car_control()
#
#     def midpoint_calculate_angle(self, point):
#         x = point[0]
#         y = point[1]
#         dx = float(x - self.carPos_x)
#         dy = float(self.carPos_y - y)
#         if dy == 0:
#             return 0
#         if dx < 0:
#             angle = -math.atan(-dx / dy) * 180 / np.pi
#         else:
#             angle = math.atan(dx / dy) * 180 / np.pi
#
#         # print('angle from midpoint: ', angle)
#         return angle
#
#     def findCenter(self):
#         avg_midpoint = [120, 0]
#
#         list_cx = []
#         list_cy = []
#
#         n = 0
#         # list_center = []
#         for i in range(0, 300, 10):
#             # pd.isNearCross = False
#             layer = self.road[i:i + 10]
#             _, contours, _ = cv.findContours(layer, cv.RETR_TREE,
#                                              cv.CHAIN_APPROX_SIMPLE)
#             if len(contours) == 1:
#                 M = cv.moments(contours[0])
#                 # print("m00",M['m00'])
#                 if M['m00'] < 700:
#                     try:
#                         cx = int(M['m10'] / M['m00'])
#                         cy = int(M['m01'] / M['m00']) + i
#
#                         midpoint = (cx, cy)
#                         # list_center.append([cx, cy])
#                         list_cx.append(cx)
#                         list_cy.append(cy)
#                         cv.circle(self.center, (midpoint), 3, (255, 255, 255), -1)
#                         n += 1
#                     except ZeroDivisionError:
#                         pass
#                 # if M['m00'] > 400:
#                 #     print("dmm co giam toc ko?")
#                 #     pd.isNearCross = True
#
#         partx = list_cx[n / 2:5 * n / 6]
#         party = list_cy[n / 2:5 * n / 6]
#
#         for i in range(len(partx)):
#             cv.circle(self.center, tuple([partx[i], party[i]]), 3, (0, 0, 255),
#                       -1)
#         try:
#             avg_midpoint = [sum(partx) / len(partx), sum(party) / len(party)]
#         except ZeroDivisionError:
#             pass
#
#         if pd.obstacle_left != 0 and pd.obstacle_left[2] * pd.obstacle_left[
#             3] > pd.thresh_modify:
#             avg_midpoint[0] += int(pd.obstacle_left[2] * pd.modify_center)
#
#         if pd.obstacle_right != 0 and pd.obstacle_right[2] * pd.obstacle_right[
#             3] > pd.thresh_modify:
#             avg_midpoint[0] -= int(pd.obstacle_right[2] * pd.modify_center)
#
#         # if pd.leftSign:
#         #     avg_midpoint[0] += 10
#         # if pd.rightSign:
#         #     avg_midpoint[0] -= 10
#
#         print('midpoint', avg_midpoint)
#
#         cv.circle(self.center, tuple(avg_midpoint), 3, (0, 255, 0), -1)
#
#         return avg_midpoint
#
#     def MoveInNormalWay(self):
#         pd.car_rotation = self.midpoint_calculate_angle(self.findCenter())
#         self.ControlSpeed()
#
#     def ControlSpeed(self):
#         # if abs(pd.car_rotation) <2:
#         #     pd.car_speed = 70
#         #     pd.goodDistance = 92
#         # else:
#         #     pd.car_speed = 60
#         #     pd.goodDistance = 82
#         if time() - pd.timeStartGame < 10:
#             pd.goodDistance = 75
#         else:
#             pd.goodDistance = 84
#
#     def CheckTimeForUsingMode1(self):
#         if pd.leftSign and pd.startRotateLeft == False:
#             print("hallllllsdfsadfsd")
#             pd.car_speed = -20  # the car move slower on crossland
#             # detect left road to rotate
#             layer_left = self.road[220:, :90]
#             cv.line(self.center, (90, 0), (90, 300), (255, 0, 0), 1)
#             _, contours, _ = cv.findContours(layer_left, cv.RETR_TREE,
#                                              cv.CHAIN_APPROX_SIMPLE)
#             if len(contours) == 1:
#                 M = cv.moments(contours[0])
#                 try:
#                     cx = int(M['m10'] / M['m00'])
#                     cy = int(M['m01'] / M['m00']) + 220
#                     area_left = cv.contourArea(contours[0])
#                     distanceToCross = 320 - cy
#                     print("cy:", cy)
#                     print("good:", pd.goodDistance)
#                     print("arealeft:", area_left)
#                     if area_left >= 25 and distanceToCross <= pd.goodDistance:
#                         pd.startRotateLeft = True
#                         pd.startTime = time()
#                         pd.currentMode = 1
#                 except ZeroDivisionError:
#                     pass
#
#         if pd.rightSign and pd.startRotateRight == False:
#             print("hallllllsdfsadfsd")
#             pd.car_speed = -20  # the car move slower on crossland
#             # detect left road to rotate
#             layer_right = self.road[220:, 150:]
#             cv.line(self.center, (150, 0), (150, 300), (255, 0, 0), 1)
#             cv.line(self.center, (0, 220), (200, 220), (255, 0, 0), 1)
#             _, contours, _ = cv.findContours(layer_right, cv.RETR_TREE,
#                                              cv.CHAIN_APPROX_SIMPLE)
#             if len(contours) == 1:
#                 M = cv.moments(contours[0])
#                 try:
#                     cx = int(M['m10'] / M['m00'])
#                     cy = int(M['m01'] / M['m00']) + 220
#                     area_right = cv.contourArea(contours[0])
#                     distanceToCross = 320 - cy
#                     print("cy:", cy)
#                     print("good:", pd.goodDistance)
#                     print("arearigt:", area_right)
#                     if area_right >= 25 and distanceToCross <= pd.goodDistance:
#                         pd.startRotateRight = True
#                         pd.startTime = time()
#                         pd.currentMode = 1
#                 except ZeroDivisionError:
#                     pass
#
#     def RotateOnCrossLand(self):
#         if pd.startRotateLeft:
#             layer_left = self.road[:, :85]
#             pd.car_rotation = -17
#             _, contours, _ = cv.findContours(layer_left, cv.RETR_TREE,
#                                              cv.CHAIN_APPROX_SIMPLE)
#             # stop rotate when detect new road
#             if len(contours) == 1:
#                 area_left1 = cv.contourArea(contours[0])
#                 print("area:", area_left1)
#                 if area_left1 >= 2500:
#                     print("currentmode:", pd.currentMode)
#                     pd.leftSign = False
#                     pd.currentMode = 0
#                     pd.startRotateLeft = False
#                     pd.car_speed = 60
#         if pd.startRotateRight:
#             layer_right = self.road[:, 155:]
#             pd.car_rotation = 17
#             _, contours, _ = cv.findContours(layer_right, cv.RETR_TREE,
#                                              cv.CHAIN_APPROX_SIMPLE)
#             # stop rotate when detect new road
#             if len(contours) == 1:
#                 area_right1 = cv.contourArea(contours[0])
#                 print("area:", area_right1)
#                 if area_right1 >= 2500:
#                     pd.rightSign = False
#                     print("currentmode:", pd.currentMode)
#                     pd.currentMode = 0
#                     pd.startRotateRight = False
#                     pd.car_speed = 60
#
#     def car_control(self):
#         # avg_midpoint = self.findCenter()
#         # if pd.hasSign == True:
#         #     self.center_vertical_layers()
#
#         if time() - pd.startTime > 10 and pd.currentMode == 1:
#             pd.rightSign = False
#             pd.leftSign = False
#             pd.currentMode = 0
#             pd.startRotateRight = False
#             pd.startRotateLeft = False
#             pd.car_speed = 60
#             pd.startTime = 0
#         self.CheckTimeForUsingMode1()
#         if pd.currentMode == 0:
#             self.MoveInNormalWay()
#         if pd.currentMode == 1:
#             self.RotateOnCrossLand()
#         print("mode:", pd.currentMode)
#         print("angle", pd.car_rotation)
#         steer_publisher.publish(pd.car_rotation)
#         speed_publisher.publish(pd.car_speed)

import asyncio
import base64
import json
import time
from io import BytesIO
from multiprocessing import Process, Queue

import cv2
import numpy as np
import websockets
from PIL import Image
from simple_pid import PID

from lane_line_detection import calculate_control_signal
from traffic_sign_detection import detect_traffic_signs


# Initalize traffic sign classifier
traffic_sign_model = cv2.dnn.readNetFromONNX(
    "traffic_sign_classifier_lenet_v3.onnx")

# Global queue to save current image
# We need to run the sign classification model in a separate process
# Use this queue as an intermediate place to exchange images
g_image_queue = Queue(maxsize=5)
g_sign_queue = Queue(maxsize=10)

count = 0
# Function to run sign classification model continuously
# We will start a new process for this
def process_traffic_sign_loop(g_image_queue):
    while True:
        if g_image_queue.empty():
            time.sleep(0.1)
            continue

        image = g_image_queue.get()

        # Prepare visualization image
        draw = image.copy()
        # Detect traffic signs
        signs = detect_traffic_signs(image, traffic_sign_model, draw=draw)
        if not g_sign_queue.full():
            if len(signs) > 0:
                g_sign_queue.put(signs[-1][0])

        # Show the result to a window
        cv2.imshow("Traffic signs", draw)
        cv2.waitKey(1)



async def process_image(websocket, path):
    async for message in websocket:
        # Get image from simulation
        data = json.loads(message)
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        global count
        count += 1
        if(count % 5 == 0):
            id = count/5
            direct = '/home/ankhanh/CuocDuaSo/data/capture03/pic' + str(id) + '.jpg'
            cv2.imwrite(direct, image)

        image = cv2.resize(image, (640, 480))

        # Prepare visualization image
        draw = image.copy()

        # get sign
        cur_sign = None
        if not g_sign_queue.empty():
            cur_sign = g_sign_queue.get()

        # Send back throttle and steering angle
        throttle, steering_angle = calculate_control_signal(image, cur_sign, draw=draw)
        # throttle, steering_angle = 0, 0
        # Update image to g_image_queue - used to run sign detection
        if not g_image_queue.full():
            g_image_queue.put(image)

        # Show the result to a window
        cv2.imshow("Result", draw)
        cv2.waitKey(1)

        # Send back throttle and steering angle
        message = json.dumps(
            {"throttle": throttle, "steering": steering_angle})
        print(message)
        await websocket.send(message)


async def main():
    async with websockets.serve(process_image, "0.0.0.0", 4567, ping_interval=None):
        await asyncio.Future()  # run forever

if __name__ == '__main__':
    p = Process(target=process_traffic_sign_loop, args=(g_image_queue,))
    p.start()
    asyncio.run(main())