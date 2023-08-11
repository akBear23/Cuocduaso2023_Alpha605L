'''
segment the lane and get birdview of that segment
'''

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
# import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
import cv2
# model = load_model('/home/ankhanh/CuocDuaSo/models/final-road-seg-model-v3.h5')
model = load_model('via_road_seg_v3.h5')
# model = load_model('ENET_20191227.h5')
img_path = '/home/ankhanh/CuocDuaSo/data/fetch/pic676.jpg'
# img_path = '/home/ankhanh/CuocDuaSo/models/cds2020/data/lane_line_images/364811203_302588935628685_6026889737141278233_n.png'
# img_path = '02_00_000.png'
img = cv2.imread(img_path)

def find_lane_lines(birdview_mask):
    """
    Detecting road markings
    This function will take a birdview mask,
    Returns a filtered image of road markings
    """
    lines = cv2.HoughLinesP(birdview_mask, rho=1, theta=np.pi/180, threshold=20, minLineLength=20, maxLineGap=300)
    left_lines = []
    right_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 == 0:
            continue  # Skip vertical lines to avoid division by zero
        slope = (y2 - y1) / (x2 - x1)
        length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        if slope < -0.5 and length > 100:
            left_lines.append(line)
        elif slope > 0.5 and length > 100:
            right_lines.append(line)

    left_line = np.mean(left_lines, axis=0, dtype=np.int32)
    right_line = np.mean(right_lines, axis=0, dtype=np.int32)
    # line_image = np.zeros_like(birdview_mask, dtype=np.uint8)

    # line_image = np.zeros((120, 160, 3), dtype = np.uint8)
    line_image = birdview_mask.copy()
    cv2.line(line_image, (left_line[0][0], left_line[0][1]), (left_line[0][2], left_line[0][3]), (0, 0, 255), 5)
    cv2.line(line_image, (right_line[0][0], right_line[0][1]), (right_line[0][2], right_line[0][3]), (0, 0, 255), 5)

    # weighted_mask = np.expand_dims(birdview_mask, axis = -1)
    # weighted_mask = np.repeat(weighted_mask, 3, axis=-1)
    # print(weighted_mask.shape)

    # lane_lines = cv2.bitwise_or(line_image, weighted_mask)
    cv2.imshow('overlay',line_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # print(left_line, right_line)

def birdview_transform(mask):
    """Apply bird-view transform to the image
    """

    src_points = np.float32([[0, 100], [160, 100], [0, 120], [160, 120]])

    # Define the destination points for the bird's eye view transform
    dst_points = np.float32([[0, 0], [160, 0], [0, 120], [160, 120]])

    # Perform the perspective transform
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the bird's eye view transform to the mask
    birdview_mask = cv2.warpPerspective(mask, M, (160, 120))

    # Display the original mask and the bird's eye view mask
    # cv2.imshow("Original Mask", mask)
    # cv2.imshow("Bird's Eye View Mask", birdview_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return birdview_mask


height, width = 120, 160
# input is image
def resize_img_2(img):
    #img = cv2.imread(img_path)
    img = cv2.resize(img, (width, height))
    #plt.imshow(img)
    return img

def predict_2(img, model):
    img = resize_img_2(img)
    pred = model.predict(np.expand_dims(img, 0))
    pred = np.where(pred>=0.5, 1.0, 0.0)
    pred = pred.reshape(120, 160)
    plt.imshow(pred)
    return pred
'''
pred_img = predict_2(img)
mask = np.array(pred_img)
'''
def clean_img(mask):
    mask = mask.astype(np.uint8)
    mask_normalized = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX)
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area or other properties
    min_area_threshold = 450  # Adjust the minimum area threshold as needed
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area_threshold]

    # Create a new mask with filtered contours
    cleaned_mask = np.zeros_like(mask)
    cv2.drawContours(cleaned_mask, filtered_contours, -1, 255, thickness=cv2.FILLED)
    cleaned_mask = np.squeeze(cleaned_mask)

    # cv2.imshow('Original Mask', mask_normalized)
    # cv2.imshow('Cleaned Mask', cleaned_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return cleaned_mask

'''
cleaned_mask = clean_img(mask)
cleaned_mask = cv2.normalize(cleaned_mask, None, 0, 255, cv2.NORM_MINMAX)
birdview_mask = birdview_transform(cleaned_mask)
'''

# find_lane_lines(birdview_mask)
#another approach
# def denoiseLane(img):
#     kernel = np.ones((10, 10), np.uint8)
#     erosion = cv2.erode(img, kernel, iterations=1)
#     return erosion
# x = denoiseLane(mask)
# cv2.imshow('denoised', x)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

def measureLane(img):
    model = load_model('via_road_seg_v3.h5')
    pred_img = predict_2(img, model)
    mask = np.array(pred_img)
    cleaned_mask = clean_img(mask)
    mask_birdv = birdview_transform(cleaned_mask)
    ls_cx = []
    ls_cy = []

    centroids = cv2.cvtColor((mask_birdv), cv2.COLOR_GRAY2BGR)

    for i in range(0, 120, 10):
        ls_m10 = []
        ls_m01 = []
        ls_m00 = []

        layer = mask_birdv[i:i+10]
        contours, hierarchy = cv2.findContours(layer, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            M = cv2.moments(cnt)
            ls_m10.append(M['m10'])
            ls_m01.append(M['m01'])
            ls_m00.append(M['m00'])

        total_area = sum(ls_m00)
        # print('area', total_area)
        if 100 < total_area < 1200:
            try:
                mean_cx = int(sum(ls_m10)/total_area)
                mean_cy = int(sum(ls_m01)/total_area) + i
                cv2.circle(centroids, (mean_cx, mean_cy), 3, (0, 0, 255), -1)
                ls_cx.append(mean_cx)
                ls_cy.append(mean_cy)

            except ZeroDivisionError:
                pass

    cv2.imshow('centroids', centroids)
    cv2.waitKey(1)
    # cv2.destroyAllWindows()
    # print(ls_cx, ls_cy)
    return ls_cx, ls_cy

measureLane(img)





