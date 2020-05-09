# coding: utf-8
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import cv2
from keras.models import load_model
import time

# model_smoke_detector = load_model('model_smoke_detector_vgg16.h5')
#
# model_vgg16 = VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
# # cap = cv2.VideoCapture(r'H:\研究生课题\DataSet\POLARBEARWYYSmoke-Detection\video_test_set\test-6.avi')
# # cap = cv2.VideoCapture(r'H:\研究生课题\DataSet\POLARBEARWYYSmoke-Detection\video_train_set\train-13.avi')
# # cap = cv2.VideoCapture(r'H:\研究生课题\DataSet\yuanfeiniu\smokevideos\Dry_leaf_smoke_02.avi')
# # cap = cv2.VideoCapture(r'H:\研究生课题\DataSet\yuanfeiniu\smokevideos\Cotton_rope_smoke_04.avi')
# cap = cv2.VideoCapture(r'H:\研究生课题\DataSet\0124Video_Smoke_Detection\dataset\部分测试视频\21.mpg')
#
# video_writer = cv2.VideoWriter('smoke_detection_output2.avi', cv2.VideoWriter_fourcc(*'MJPG'), 24, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
#
# steps = [5, 150, 500]
# # frames = [0] * 30
# choice = 0
# total_time = 0
# count = 0
# # num = 1
# while True:
#     status, img = cap.read()
#     print(status)
img = '000001.jpg'
img_detect = cv2.resize(img[:, :, [2, 1, 0]], (224, 224))
img = cv2.flip(img, 1)
cv2.imshow('daf', img_detect)