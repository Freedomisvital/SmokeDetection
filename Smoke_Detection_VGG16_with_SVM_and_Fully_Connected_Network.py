# coding: utf-8
import numpy as np
from sklearn import svm
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.mobilenet import MobileNet
# from keras.applications.mobilenet import preprocess_input
# # 读取VGG16预训练模型参数
model_vgg16 = VGG16(weights='imagenet', include_top=False)
# model_vgg16 = MobileNet(input_shape=(224, 224, 3), weights='imagenet', include_top=False)
# # 读取训练集图片计算特征

training_set_vgg16_features = []
for i in list(range(1, 5001)):
    img_path = r'H:\研究生课题\DataSet\POLARBEARWYYSmoke-Detection\test8\smoke\%06d.jpg' % i
    img = image.load_img(img_path, target_size=(224, 224))# 读取图片
    x = image.img_to_array(img)# 图片转换为ndarray
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)# 数组转换为vgg16输入格式

    training_set_vgg16_features.append(model_vgg16.predict(x).reshape((7*7*512,)))# 计算该张图片的特征

for i in list(range(1, 5001)):
    img_path = r'H:\研究生课题\DataSet\POLARBEARWYYSmoke-Detection\test8\nosmoke\%06d.jpg' % i
    img = image.load_img(img_path, target_size=(224, 224))# 读取图片
    x = image.img_to_array(img)# 图片转换为ndarray
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)# 数组转换为vgg16输入格式

    training_set_vgg16_features.append(model_vgg16.predict(x).reshape((7*7*512,)))# 计算该张图片的特征



training_set_vgg16_features_ndarray = np.vstack(training_set_vgg16_features)# 转换为ndarray 转换为一列

training_set_label = np.array([1.0 if i < 5000 else 0.0 for i in range(10000)])


# # 读取测试集图片计算特征
test_set_vgg16_features = []

for i in list(range(5001, 6001)):
    img_path = r'H:\研究生课题\DataSet\POLARBEARWYYSmoke-Detection\test8\smoke\%06d.jpg' % i
    img = image.load_img(img_path, target_size=(224, 224))# 读取图片
    x = image.img_to_array(img)# 图片转换为ndarray
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)# 数组转换为vgg16输入格式

    test_set_vgg16_features.append(model_vgg16.predict(x).reshape((7*7*512, )))# 计算该张图片的特征

for i in list(range(5001, 6001)):
    img_path = r'H:\研究生课题\DataSet\POLARBEARWYYSmoke-Detection\test8\nosmoke\%06d.jpg' % i
    img = image.load_img(img_path, target_size=(224, 224))# 读取图片
    x = image.img_to_array(img)# 图片转换为ndarray
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)# 数组转换为vgg16输入格式

    test_set_vgg16_features.append(model_vgg16.predict(x).reshape((7*7*512, )))# 计算该张图片的特征


test_set_vgg16_features_ndarray = np.vstack(test_set_vgg16_features)# 转换为ndarray

# # VGG16+SVM

# ### 定义SVM
clf_vgg = svm.SVC()

# ### 训练SVM
clf_vgg.fit(training_set_vgg16_features_ndarray, training_set_label)

# ### 计算SVM在训练集上的预测值
training_set_vgg16_with_svm_prediction = clf_vgg.predict(training_set_vgg16_features_ndarray)

training_set_vgg16_with_svm_prediction[training_set_vgg16_with_svm_prediction >= 0.5] = 1
training_set_vgg16_with_svm_prediction[training_set_vgg16_with_svm_prediction < 0.5] = 0

training_set_vgg16_with_svm_prediction_positive = training_set_vgg16_with_svm_prediction[: 5000]
training_set_vgg16_with_svm_prediction_negative = training_set_vgg16_with_svm_prediction[5000:]


# ### 计算训练集准确率

print('VGG16+SVM Training Set Confusion Matrix\n          Ture      False')

print('Positive： {0}        {1}'.format(
    len(training_set_vgg16_with_svm_prediction_positive[training_set_vgg16_with_svm_prediction_positive == 1]),len(training_set_vgg16_with_svm_prediction_positive[training_set_vgg16_with_svm_prediction_positive == 0])))
print('Negative： {0}        {1}'.format(
    len(training_set_vgg16_with_svm_prediction_negative[training_set_vgg16_with_svm_prediction_negative == 0]),len(training_set_vgg16_with_svm_prediction_negative[training_set_vgg16_with_svm_prediction_negative == 1])))
print('检测率: {0}, 虚警率: {1}'.format(
    (len(training_set_vgg16_with_svm_prediction_positive[training_set_vgg16_with_svm_prediction_positive == 1]) + 0.0) / len(training_set_vgg16_with_svm_prediction_positive),
    (len(training_set_vgg16_with_svm_prediction_negative[training_set_vgg16_with_svm_prediction_negative == 1]) + 0.0) / len(training_set_vgg16_with_svm_prediction_negative)
))


# ### 计算SVM在测试集上的预测值
test_set_vgg16_with_svm_prediction = clf_vgg.predict(test_set_vgg16_features_ndarray)

test_set_vgg16_with_svm_prediction[test_set_vgg16_with_svm_prediction >= 0.5] = 1
test_set_vgg16_with_svm_prediction[test_set_vgg16_with_svm_prediction < 0.5] = 0
test_set_vgg16_with_svm_prediction_positive = test_set_vgg16_with_svm_prediction[: 1000]
test_set_vgg16_with_svm_prediction_negative = test_set_vgg16_with_svm_prediction[1000:]


# ### 计算测试集准确率

print('VGG16+SVM test Set Confusion Matrix\n          Ture      False')

print('Positive： {0}        {1}'.format(
    len(test_set_vgg16_with_svm_prediction_positive[test_set_vgg16_with_svm_prediction_positive == 1]),len(test_set_vgg16_with_svm_prediction_positive[test_set_vgg16_with_svm_prediction_positive == 0])))
print('Negative： {0}        {1}'.format(
    len(test_set_vgg16_with_svm_prediction_negative[test_set_vgg16_with_svm_prediction_negative == 0]),len(test_set_vgg16_with_svm_prediction_negative[test_set_vgg16_with_svm_prediction_negative == 1])))
print('检测率: {0}, 虚警率: {1}'.format(
    (len(test_set_vgg16_with_svm_prediction_positive[test_set_vgg16_with_svm_prediction_positive == 1]) + 0.0) / len(test_set_vgg16_with_svm_prediction_positive),
    (len(test_set_vgg16_with_svm_prediction_negative[test_set_vgg16_with_svm_prediction_negative == 1]) + 0.0) / len(test_set_vgg16_with_svm_prediction_negative)
))

## 全连接神经网络

#### 导入相关库
from keras.models import Sequential
from keras.layers import Activation, Dropout, Dense

# ### 定义全连接神经网络
model_smoke_detector = Sequential()
model_smoke_detector.add(Dense(1024, activation='sigmoid', input_shape=(7*7*512, )))# 加入全连接层
model_smoke_detector.add(Dropout(0.5))# 加入dropout防止过拟合
model_smoke_detector.add(Dense(128, activation='sigmoid'))# 加入全连接层
model_smoke_detector.add(Dropout(0.5))# 加入dropout防止过拟合
model_smoke_detector.add(Dense(1, activation='sigmoid'))# 加入全连接层
# 定义神经网络损失函数等
model_smoke_detector.compile(optimizer='adam',
                             loss='binary_crossentropy',
                             metrics=['accuracy'])

# 拟合数据即训练神经网络
model_smoke_detector.fit(training_set_vgg16_features_ndarray, training_set_label, epochs=10, batch_size=16)

# 对测试集进行预测
test_set_vgg16_with_fc_prediction = model_smoke_detector.predict(test_set_vgg16_features_ndarray)
# 对训练集进行预测
training_set_vgg16_with_fc_prediction = model_smoke_detector.predict(training_set_vgg16_features_ndarray)
# In[ ]:
test_set_vgg16_with_fc_prediction[test_set_vgg16_with_fc_prediction >= 0.5] = 1
test_set_vgg16_with_fc_prediction[test_set_vgg16_with_fc_prediction < 0.5] = 0
test_set_vgg16_with_fc_prediction_positive = test_set_vgg16_with_fc_prediction[:1000]
test_set_vgg16_with_fc_prediction_negative = test_set_vgg16_with_fc_prediction[1000:]


training_set_vgg16_with_fc_prediction[training_set_vgg16_with_fc_prediction >= 0.5] = 1
training_set_vgg16_with_fc_prediction[training_set_vgg16_with_fc_prediction < 0.5] = 0
training_set_vgg16_with_fc_prediction_positive = training_set_vgg16_with_fc_prediction[:5000]
training_set_vgg16_with_fc_prediction_negative = training_set_vgg16_with_fc_prediction[5000:]

# VGG16 + FC Training Set confusion Matrix
print('VGG16+FC Training Set Confusion Matrix\n          Ture      False')

print('Positive： {0}        {1}'.format(
    len(training_set_vgg16_with_fc_prediction_positive[training_set_vgg16_with_fc_prediction_positive == 1]),len(training_set_vgg16_with_fc_prediction_positive[training_set_vgg16_with_fc_prediction_positive == 0])))
print('Negative： {0}        {1}'.format(
    len(training_set_vgg16_with_fc_prediction_negative[training_set_vgg16_with_fc_prediction_negative == 0]),len(training_set_vgg16_with_fc_prediction_negative[training_set_vgg16_with_fc_prediction_negative == 1])))
print('检测率: {0}, 虚警率: {1}'.format(
    (len(training_set_vgg16_with_fc_prediction_positive[training_set_vgg16_with_fc_prediction_positive == 1]) + 0.0) / len(training_set_vgg16_with_fc_prediction_positive),
    (len(training_set_vgg16_with_fc_prediction_negative[training_set_vgg16_with_fc_prediction_negative == 1]) + 0.0) / len(training_set_vgg16_with_fc_prediction_negative)
))

# VGG16 + FC test Set Confusion Matrix
print('VGG16+FC test Set Confusion Matrix\n          Ture      False')

print('Positive： {0}        {1}'.format(
    len(test_set_vgg16_with_fc_prediction_positive[test_set_vgg16_with_fc_prediction_positive == 1]),len(test_set_vgg16_with_fc_prediction_positive[test_set_vgg16_with_fc_prediction_positive == 0])))
print('Negative： {0}        {1}'.format(
    len(test_set_vgg16_with_fc_prediction_negative[test_set_vgg16_with_fc_prediction_negative == 0]),len(test_set_vgg16_with_fc_prediction_negative[test_set_vgg16_with_fc_prediction_negative == 1])))
print('检测率: {0}, 虚警率: {1}'.format(
    (len(test_set_vgg16_with_fc_prediction_positive[test_set_vgg16_with_fc_prediction_positive == 1]) + 0.0) / len(test_set_vgg16_with_fc_prediction_positive),
    (len(test_set_vgg16_with_fc_prediction_negative[test_set_vgg16_with_fc_prediction_negative == 1]) + 0.0) / len(test_set_vgg16_with_fc_prediction_negative)
))

# 保存模型
model_smoke_detector.save('b1_model_smoke_detector_vgg16.h5')


