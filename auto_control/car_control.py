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
from keras.models import load_model
from simple_pid import PID

from lane_line_detection import calculate_control_signal
from traffic_sign_detection import detect_traffic_signs
from lane_segment import measureLane

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
        measureLane(image)
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