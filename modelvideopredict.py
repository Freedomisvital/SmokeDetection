# coding: utf-8
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import cv2
from keras.models import load_model
import time

model_smoke_detector = load_model('model_smoke_detector_vgg16.h5')

model_vgg16 = VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
# cap = cv2.VideoCapture(r'H:\研究生课题\DataSet\POLARBEARWYYSmoke-Detection\video_test_set\test-6.avi')
# cap = cv2.VideoCapture(r'H:\研究生课题\DataSet\POLARBEARWYYSmoke-Detection\video_train_set\train-13.avi')
# cap = cv2.VideoCapture(r'H:\研究生课题\DataSet\yuanfeiniu\smokevideos\Dry_leaf_smoke_02.avi')
# cap = cv2.VideoCapture(r'H:\研究生课题\DataSet\yuanfeiniu\smokevideos\Cotton_rope_smoke_04.avi')
# cap = cv2.VideoCapture(r'H:\研究生课题\DataSet\0124Video_Smoke_Detection\dataset\部分测试视频\21.mpg')
# cap = cv2.VideoCapture(r'H:\研究生课题\DataSet\POLARBEARWYYSmoke-Detection\video_train_set\train-1.avi')
cap = cv2.VideoCapture(r'000001.jpg')
# cap = cv2.VideoCapture('smoke2.mp4')
video_writer = cv2.VideoWriter('smoke_detection_output_b0.avi', cv2.VideoWriter_fourcc(*'MJPG'), 24, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

steps = [5, 150, 500]
# frames = [0] * 30
choice = 0
total_time = 0
count = 0
error_count = 0
# num = 1
while True:
    status, img = cap.read()
    # img = img[:, ::-1, :]
    #img 代表截取到的一帧图片
    if status:   #第一个参数status为True或者False,代表有没有读取到图片
        img_detect = cv2.resize(img[:, :, [2, 1, 0]], (224, 224))
        start_time = time.time()
        x = image.img_to_array(img_detect)
        x = np.expand_dims(x, axis=0) #
        x = preprocess_input(x)
        feature = model_vgg16.predict(x).reshape((1, 7* 7 * 512)) #对读入的图片进行预测
        result = model_smoke_detector.predict(feature)[0, 0] #预测结果,是一个小数
        if result < 0.5:
            error_count += error_count
        end_time = time.time()

        total_time += end_time - start_time
        count += 1
        #Probability
        # frames.pop(0)
        # frames.append(1 if result > 0.5 else 0)
        img1 = cv2.flip(img, 1) #水平翻转图像
        cv2.putText(img1,
                    'P: {0}'.format(str(result)), #添加的数字
                    (30, 30), #左上角坐标
                    cv2.FONT_HERSHEY_COMPLEX, #字体
                    1,        #字体大小
                    (0, 0, 255), #颜色
                    1)           #字体粗细
        cv2.putText(img1,
                    '{0}'.format('smoke' if result > 0.5 else 'no smoke'),
                    (30, 90),
                    cv2.FONT_HERSHEY_COMPLEX,
                    2,
                    (0, 0, 255),
                    2)
        '''cv2.putText(img,
                    '{0}'.format('smoke' if sum(frames) > 0 else 'no smoke'),
                    (30, 90), 
                    cv2.FONT_HERSHEY_COMPLEX, 
                    2, 
                    (0, 0, 255), 
                    2)'''
        cv2.imshow('', img1)
        key = cv2.waitKey(steps[choice])
        video_writer.write(img1)
        if key == 27:
            break
        elif key == ord('n'): # 110
            choice += 1
            choice %= 3
    else:
        cv2.destroyAllWindows() #表示结束
        break

print('预测总时间：', total_time, '(s)')
print('视频帧的总数量：', count, '(s)')
print('错误帧数：', error_count, '(s)')
cap.release()
video_writer.release()

# total_time / count
