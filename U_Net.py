# %%
from sklearn.model_selection import train_test_split
import os
import glob
import cv2
import numpy as np  # linear algebra
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.util import montage

montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
data_dir = 'dataset'
train_image_dir = os.path.join(data_dir, 'train')
test_image_dir = os.path.join(data_dir, 'test')
import gc;

gc.enable()  # memory is tight
from skimage.morphology import label

# %%
BATCH_SIZE = 4
EDGE_CROP = 2
NB_EPOCHS = 20
GAUSSIAN_NOISE = 0.1
UPSAMPLE_MODE = 'DECONV'
# downsampling inside the network
NET_SCALING = None
# downsampling in preprocessing
IMG_SCALING = (1, 1)
# number of validation images to use
VALID_IMG_COUNT = 400
# maximum number of steps_per_epoch in training
MAX_TRAIN_STEPS = 200
AUGMENT_BRIGHTNESS = False


# %%
def get_all_imgs():
    img_path = os.path.join(train_image_dir, 'images')
    images = glob.glob(os.path.join(img_path, '*.*'))
    return [os.path.basename(image) for image in images]


# %%
# print(get_all_imgs())
TRAIN_IMGS, TEST_IMGS = train_test_split(get_all_imgs())
# %%
import random


def cv2_brightness_augment(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]
    seed = random.uniform(0.5, 1.2)
    v = ((v / 255.0) * seed) * 255.0
    hsv[:, :, 2] = np.array(np.clip(v, 0, 255), dtype=np.uint8)
    rgb_final = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb_final


# %%
def make_image_gen(img_file_list=TRAIN_IMGS, batch_size=BATCH_SIZE):
    all_batches = TRAIN_IMGS
    out_rgb = []
    out_mask = []
    img_path = os.path.join(train_image_dir, 'images')
    mask_path = os.path.join(train_image_dir, 'masks')
    while True:
        np.random.shuffle(all_batches)
        for c_img_id in all_batches:
            c_img = imread(os.path.join(img_path, c_img_id))
            c_img = cv2_brightness_augment(c_img)
            c_mask = imread(os.path.join(mask_path, c_img_id))
            if IMG_SCALING is not None:
                c_img = cv2.resize(c_img, (256, 256), interpolation=cv2.INTER_AREA)
                c_mask = cv2.resize(c_mask, (256, 256), interpolation=cv2.INTER_AREA)
            c_mask = np.reshape(c_mask, (c_mask.shape[0], c_mask.shape[1], -1))
            c_mask = c_mask > 0
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb) >= batch_size:
                yield np.stack(out_rgb, 0) / 255.0, np.stack(out_mask, 0)
                out_rgb, out_mask = [], []


# %% md
## Make Training Set
# %%
train_gen = make_image_gen()
train_x, train_y = next(train_gen)
print('x', train_x.shape, train_x.min(), train_x.max())
print('y', train_y.shape, train_y.min(), train_y.max())
# %%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
batch_rgb = montage_rgb(train_x)
batch_seg = montage(train_y[:, :, :, 0])
ax1.imshow(batch_rgb)
ax1.set_title('Images')
ax2.imshow(batch_seg, cmap='gray')
ax2.set_title('Segmentations')
ax3.imshow(mark_boundaries(batch_rgb,
                           batch_seg.astype(int)))
ax3.set_title('Outlined Smokes')
fig.savefig('overview.png')
# %% md
## Make Validation Set
# %%
valid_x, valid_y = next(make_image_gen(TEST_IMGS, len(TEST_IMGS)))
print(valid_x.shape, valid_y.shape)
# %% md
## Augment Data
# %%
from keras.preprocessing.image import ImageDataGenerator

dg_args = dict(featurewise_center=False,
               samplewise_center=False,
               rotation_range=15,
               width_shift_range=0.1,
               height_shift_range=0.1,
               shear_range=0.01,
               zoom_range=[0.9, 1.25],
               horizontal_flip=True,
               vertical_flip=False,
               fill_mode='reflect',
               data_format='channels_last')
# brightness can be problematic since it seems to change the labels differently from the images
if AUGMENT_BRIGHTNESS:
    dg_args[' brightness_range'] = [0.5, 1.5]
image_gen = ImageDataGenerator(**dg_args)

if AUGMENT_BRIGHTNESS:
    dg_args.pop('brightness_range')
label_gen = ImageDataGenerator(**dg_args)


def create_aug_gen(in_gen, seed=None):
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_x = image_gen.flow(255 * in_x,
                             batch_size=in_x.shape[0],
                             seed=seed,
                             shuffle=True)
        g_y = label_gen.flow(in_y,
                             batch_size=in_x.shape[0],
                             seed=seed,
                             shuffle=True)

        yield next(g_x) / 255.0, next(g_y)


# %%
cur_gen = create_aug_gen(train_gen)
t_x, t_y = next(cur_gen)
print('x', t_x.shape, t_x.dtype, t_x.min(), t_x.max())
print('y', t_y.shape, t_y.dtype, t_y.min(), t_y.max())
# only keep first 9 samples to examine in detail
t_x = t_x[:9]
t_y = t_y[:9]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(montage_rgb(t_x), cmap='gray')
ax1.set_title('images')
ax2.imshow(montage(t_y[:, :, :, 0]), cmap='gray')
ax2.set_title('smoke')
fig.savefig('augmentations.png')
# %%
gc.collect()
# %% md
## Build a Model
### Here we use a slight deviation on the U-Net standard model
# %%
from keras import models, layers


def upsample_conv(filters, kernel_size, strides, padding):
    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)


def upsample_simple(filters, kernel_size, strides, padding):
    return layers.UpSampling2D(strides)


if UPSAMPLE_MODE == 'DECONV':
    upsample = upsample_conv
else:
    upsample = upsample_simple

input_img = layers.Input(t_x.shape[1:], name='RGB_Input')
pp_in_layer = input_img
if NET_SCALING is not None:
    pp_in_layer = layers.AvgPool2D(NET_SCALING)(pp_in_layer)

pp_in_layer = layers.GaussianNoise(GAUSSIAN_NOISE)(pp_in_layer)
pp_in_layer = layers.BatchNormalization()(pp_in_layer)

c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(pp_in_layer)
c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
p1 = layers.MaxPooling2D((2, 2))(c1)

c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
p2 = layers.MaxPooling2D((2, 2))(c2)

c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
p3 = layers.MaxPooling2D((2, 2))(c3)

c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p4)
c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

u6 = upsample(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = layers.concatenate([u6, c4])
c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

u7 = upsample(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = layers.concatenate([u7, c3])
c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

u8 = upsample(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = layers.concatenate([u8, c2])
c8 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u8)
c8 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c8)

u9 = upsample(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = layers.concatenate([u9, c1], axis=3)
c9 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u9)
c9 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c9)

d = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
d = layers.Cropping2D((EDGE_CROP, EDGE_CROP))(d)
d = layers.ZeroPadding2D((EDGE_CROP, EDGE_CROP))(d)
if NET_SCALING is not None:
    d = layers.UpSampling2D(NET_SCALING)(d)

seg_model = models.Model(inputs=[input_img], outputs=[d])
seg_model.summary()
# %%
import keras.backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_p_bce(in_gt, in_pred):
    return 1e-3 * binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)


def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true) * K.flatten(K.round(y_pred))) / K.sum(y_true)


seg_model.compile(optimizer=Adam(1e-4, decay=1e-6), loss=dice_p_bce,
                  metrics=[dice_coef, 'binary_accuracy', true_positive_rate])
# %%
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

weight_path = "{}_weights.best.hdf5".format('seg_model')

checkpoint = ModelCheckpoint(weight_path, monitor='val_dice_coef', verbose=1,
                             save_best_only=True, mode='max', save_weights_only=True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5,
                                   patience=3,
                                   verbose=1, mode='max', epsilon=0.0001, cooldown=2, min_lr=1e-6)
early = EarlyStopping(monitor="val_dice_coef",
                      mode="max",
                      patience=15)  # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]
# %%
step_count = min(MAX_TRAIN_STEPS, len(TRAIN_IMGS) // BATCH_SIZE)
aug_gen = create_aug_gen(make_image_gen())
val_gen = make_image_gen(TEST_IMGS, len(TEST_IMGS) // BATCH_SIZE)
loss_history = [seg_model.fit_generator(aug_gen,
                                        steps_per_epoch=step_count,
                                        epochs=NB_EPOCHS,
                                        validation_data=val_gen,
                                        validation_steps=len(TEST_IMGS) // BATCH_SIZE,
                                        callbacks=callbacks_list,
                                        workers=1  # the generator is not very thread safe
                                        )]


# %%
def show_loss(loss_history):
    epich = np.cumsum(np.concatenate(
        [np.linspace(0.5, 1, len(mh.epoch)) for mh in loss_history]))
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(22, 10))
    _ = ax1.plot(epich,
                 np.concatenate([mh.history['loss'] for mh in loss_history]),
                 'b-',
                 epich, np.concatenate(
            [mh.history['val_loss'] for mh in loss_history]), 'r-')
    ax1.legend(['Training', 'Validation'])
    ax1.set_title('Loss')

    _ = ax2.plot(epich, np.concatenate(
        [mh.history['true_positive_rate'] for mh in loss_history]), 'b-',
                 epich, np.concatenate(
            [mh.history['val_true_positive_rate'] for mh in loss_history]),
                 'r-')
    ax2.legend(['Training', 'Validation'])
    ax2.set_title('True Positive Rate\n(Positive Accuracy)')

    _ = ax3.plot(epich, np.concatenate(
        [mh.history['binary_accuracy'] for mh in loss_history]), 'b-',
                 epich, np.concatenate(
            [mh.history['val_binary_accuracy'] for mh in loss_history]),
                 'r-')
    ax3.legend(['Training', 'Validation'])
    ax3.set_title('Binary Accuracy (%)')

    _ = ax4.plot(epich, np.concatenate(
        [mh.history['dice_coef'] for mh in loss_history]), 'b-',
                 epich, np.concatenate(
            [mh.history['val_dice_coef'] for mh in loss_history]),
                 'r-')
    ax4.legend(['Training', 'Validation'])
    ax4.set_title('DICE')


show_loss(loss_history)
# %%
seg_model.load_weights(weight_path)
seg_model.save('seg_model.h5')
# %%
pred_y = seg_model.predict(valid_x)
print(pred_y.shape, pred_y.min(), pred_y.max(), pred_y.mean())
# %% md
## Prepare Full Resolution Model
# %%
# if IMG_SCALING is not None:
#     fullres_model = models.Sequential()
#     fullres_model.add(layers.AvgPool2D(IMG_SCALING, input_shape = (None, None, 3)))
#     fullres_model.add(seg_model)
#     fullres_model.add(layers.UpSampling2D(IMG_SCALING))
# else:
#     fullres_model = seg_model
# fullres_model.save('fullres_model.h5')
# %% md
## Run the test data
# %%
test_paths = os.listdir(test_image_dir)
print(len(test_paths), 'test images found')
# %%
fig, m_axs = plt.subplots(20, 2, figsize=(10, 40))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
for (ax1, ax2), c_img_name in zip(m_axs, test_paths):
    c_path = os.path.join(test_image_dir, c_img_name)
    c_img = imread(c_path)
    c_img = cv2.resize(c_img, (256, 256))
    first_img = np.expand_dims(c_img, 0) / 255.0
    first_seg = seg_model.predict(first_img)
    first_img[0][:, :, 0] = (first_img[0][:, :, 0] * 0.7 + 0.5 * first_seg[0, :, :, 0])
    result = np.array(np.clip(first_img[0] * 255., 0, 255), dtype=np.int32)
    ax1.imshow(result)
    ax1.set_title('Image')
    ax2.imshow(first_seg[0, :, :, 0], vmin=0, vmax=1, cmap='gray')
    ax2.set_title('Prediction')
fig.savefig('test_predictions.png')
# %%
from moviepy.editor import VideoFileClip
def process_image(image):
   # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    image_shape = image.shape[:2]
    #     print(image_shape)
    image = cv2.resize(image, (256, 256))
    first_img = np.expand_dims(image, 0) / 255.0
    #     result = image_pipeline(image)
    first_seg = seg_model.predict(first_img)
    first_img[0][:, :, 0] = first_img[0][:, :, 0] * 0.7 + 0.3 * first_seg[0, :, :, 0]
    result = np.array(np.clip(first_img[0] * 255, 0, 255), dtype=np.float)
    #     print(image_shape[:2],result.shape,type(result[0][0][0]))
    result = cv2.resize(result, image_shape[::-1])
    #     result = result[...,::-1]
    return result


# %%
from sklearn.model_selection import train_test_split
import os
import glob
import cv2
import numpy as np  # linear algebra
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.util import montage

montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
data_dir = 'dataset'
train_image_dir = os.path.join(data_dir, 'train')
test_image_dir = os.path.join(data_dir, 'test')
import gc;

gc.enable()  # memory is tight
from skimage.morphology import label

# %%
BATCH_SIZE = 4
EDGE_CROP = 2
NB_EPOCHS = 20
GAUSSIAN_NOISE = 0.1
UPSAMPLE_MODE = 'DECONV'
# downsampling inside the network
NET_SCALING = None
# downsampling in preprocessing
IMG_SCALING = (1, 1)
# number of validation images to use
VALID_IMG_COUNT = 400
# maximum number of steps_per_epoch in training
MAX_TRAIN_STEPS = 200
AUGMENT_BRIGHTNESS = False


# %%
def get_all_imgs():
    img_path = os.path.join(train_image_dir, 'images')
    images = glob.glob(os.path.join(img_path, '*.*'))
    return [os.path.basename(image) for image in images]


# %%
# print(get_all_imgs())
TRAIN_IMGS, TEST_IMGS = train_test_split(get_all_imgs())
# %%
import random


def cv2_brightness_augment(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v = hsv[:, :, 2]
    seed = random.uniform(0.5, 1.2)
    v = ((v / 255.0) * seed) * 255.0
    hsv[:, :, 2] = np.array(np.clip(v, 0, 255), dtype=np.uint8)
    rgb_final = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb_final


# %%
def make_image_gen(img_file_list=TRAIN_IMGS, batch_size=BATCH_SIZE):
    all_batches = TRAIN_IMGS
    out_rgb = []
    out_mask = []
    img_path = os.path.join(train_image_dir, 'images')
    mask_path = os.path.join(train_image_dir, 'masks')
    while True:
        np.random.shuffle(all_batches)
        for c_img_id in all_batches:
            c_img = imread(os.path.join(img_path, c_img_id))
            c_img = cv2_brightness_augment(c_img)
            c_mask = imread(os.path.join(mask_path, c_img_id))
            if IMG_SCALING is not None:
                c_img = cv2.resize(c_img, (256, 256), interpolation=cv2.INTER_AREA)
                c_mask = cv2.resize(c_mask, (256, 256), interpolation=cv2.INTER_AREA)
            c_mask = np.reshape(c_mask, (c_mask.shape[0], c_mask.shape[1], -1))
            c_mask = c_mask > 0
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb) >= batch_size:
                yield np.stack(out_rgb, 0) / 255.0, np.stack(out_mask, 0)
                out_rgb, out_mask = [], []


# %% md
## Make Training Set
# %%
train_gen = make_image_gen()
train_x, train_y = next(train_gen)
print('x', train_x.shape, train_x.min(), train_x.max())
print('y', train_y.shape, train_y.min(), train_y.max())
# %%
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(30, 10))
batch_rgb = montage_rgb(train_x)
batch_seg = montage(train_y[:, :, :, 0])
ax1.imshow(batch_rgb)
ax1.set_title('Images')
ax2.imshow(batch_seg, cmap='gray')
ax2.set_title('Segmentations')
ax3.imshow(mark_boundaries(batch_rgb,
                           batch_seg.astype(int)))
ax3.set_title('Outlined Smokes')
fig.savefig('overview.png')
# %% md
## Make Validation Set
# %%
valid_x, valid_y = next(make_image_gen(TEST_IMGS, len(TEST_IMGS)))
print(valid_x.shape, valid_y.shape)
# %% md
## Augment Data
# %%
from keras.preprocessing.image import ImageDataGenerator

dg_args = dict(featurewise_center=False,
               samplewise_center=False,
               rotation_range=15,
               width_shift_range=0.1,
               height_shift_range=0.1,
               shear_range=0.01,
               zoom_range=[0.9, 1.25],
               horizontal_flip=True,
               vertical_flip=False,
               fill_mode='reflect',
               data_format='channels_last')
# brightness can be problematic since it seems to change the labels differently from the images
if AUGMENT_BRIGHTNESS:
    dg_args[' brightness_range'] = [0.5, 1.5]
image_gen = ImageDataGenerator(**dg_args)

if AUGMENT_BRIGHTNESS:
    dg_args.pop('brightness_range')
label_gen = ImageDataGenerator(**dg_args)


def create_aug_gen(in_gen, seed=None):
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_x = image_gen.flow(255 * in_x,
                             batch_size=in_x.shape[0],
                             seed=seed,
                             shuffle=True)
        g_y = label_gen.flow(in_y,
                             batch_size=in_x.shape[0],
                             seed=seed,
                             shuffle=True)

        yield next(g_x) / 255.0, next(g_y)


# %%
cur_gen = create_aug_gen(train_gen)
t_x, t_y = next(cur_gen)
print('x', t_x.shape, t_x.dtype, t_x.min(), t_x.max())
print('y', t_y.shape, t_y.dtype, t_y.min(), t_y.max())
# only keep first 9 samples to examine in detail
t_x = t_x[:9]
t_y = t_y[:9]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(montage_rgb(t_x), cmap='gray')
ax1.set_title('images')
ax2.imshow(montage(t_y[:, :, :, 0]), cmap='gray')
ax2.set_title('smoke')
fig.savefig('augmentations.png')
# %%
gc.collect()
# %% md
## Build a Model
### Here we use a slight deviation on the U-Net standard model
# %%
from keras import models, layers


def upsample_conv(filters, kernel_size, strides, padding):
    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)


def upsample_simple(filters, kernel_size, strides, padding):
    return layers.UpSampling2D(strides)


if UPSAMPLE_MODE == 'DECONV':
    upsample = upsample_conv
else:
    upsample = upsample_simple

input_img = layers.Input(t_x.shape[1:], name='RGB_Input')
pp_in_layer = input_img
if NET_SCALING is not None:
    pp_in_layer = layers.AvgPool2D(NET_SCALING)(pp_in_layer)

pp_in_layer = layers.GaussianNoise(GAUSSIAN_NOISE)(pp_in_layer)
pp_in_layer = layers.BatchNormalization()(pp_in_layer)

c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(pp_in_layer)
c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
p1 = layers.MaxPooling2D((2, 2))(c1)

c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
p2 = layers.MaxPooling2D((2, 2))(c2)

c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
p3 = layers.MaxPooling2D((2, 2))(c3)

c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p4)
c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

u6 = upsample(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = layers.concatenate([u6, c4])
c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

u7 = upsample(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = layers.concatenate([u7, c3])
c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

u8 = upsample(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = layers.concatenate([u8, c2])
c8 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u8)
c8 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c8)

u9 = upsample(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = layers.concatenate([u9, c1], axis=3)
c9 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u9)
c9 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c9)

d = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
d = layers.Cropping2D((EDGE_CROP, EDGE_CROP))(d)
d = layers.ZeroPadding2D((EDGE_CROP, EDGE_CROP))(d)
if NET_SCALING is not None:
    d = layers.UpSampling2D(NET_SCALING)(d)

seg_model = models.Model(inputs=[input_img], outputs=[d])
seg_model.summary()
# %%
import keras.backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy


def dice_coef(y_true, y_pred, smooth=1):
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def dice_p_bce(in_gt, in_pred):
    return 1e-3 * binary_crossentropy(in_gt, in_pred) - dice_coef(in_gt, in_pred)


def true_positive_rate(y_true, y_pred):
    return K.sum(K.flatten(y_true) * K.flatten(K.round(y_pred))) / K.sum(y_true)


seg_model.compile(optimizer=Adam(1e-4, decay=1e-6), loss=dice_p_bce,
                  metrics=[dice_coef, 'binary_accuracy', true_positive_rate])
# %%
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

weight_path = "{}_weights.best.hdf5".format('seg_model')

checkpoint = ModelCheckpoint(weight_path, monitor='val_dice_coef', verbose=1,
                             save_best_only=True, mode='max', save_weights_only=True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5,
                                   patience=3,
                                   verbose=1, mode='max', epsilon=0.0001, cooldown=2, min_lr=1e-6)
early = EarlyStopping(monitor="val_dice_coef",
                      mode="max",
                      patience=15)  # probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint, early, reduceLROnPlat]
# %%
step_count = min(MAX_TRAIN_STEPS, len(TRAIN_IMGS) // BATCH_SIZE)
aug_gen = create_aug_gen(make_image_gen())
val_gen = make_image_gen(TEST_IMGS, len(TEST_IMGS) // BATCH_SIZE)
loss_history = [seg_model.fit_generator(aug_gen,
                                        steps_per_epoch=step_count,
                                        epochs=NB_EPOCHS,
                                        validation_data=val_gen,
                                        validation_steps=len(TEST_IMGS) // BATCH_SIZE,
                                        callbacks=callbacks_list,
                                        workers=1  # the generator is not very thread safe
                                        )]


# %%
def show_loss(loss_history):
    epich = np.cumsum(np.concatenate(
        [np.linspace(0.5, 1, len(mh.epoch)) for mh in loss_history]))
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(22, 10))
    _ = ax1.plot(epich,
                 np.concatenate([mh.history['loss'] for mh in loss_history]),
                 'b-',
                 epich, np.concatenate(
            [mh.history['val_loss'] for mh in loss_history]), 'r-')
    ax1.legend(['Training', 'Validation'])
    ax1.set_title('Loss')

    _ = ax2.plot(epich, np.concatenate(
        [mh.history['true_positive_rate'] for mh in loss_history]), 'b-',
                 epich, np.concatenate(
            [mh.history['val_true_positive_rate'] for mh in loss_history]),
                 'r-')
    ax2.legend(['Training', 'Validation'])
    ax2.set_title('True Positive Rate\n(Positive Accuracy)')

    _ = ax3.plot(epich, np.concatenate(
        [mh.history['binary_accuracy'] for mh in loss_history]), 'b-',
                 epich, np.concatenate(
            [mh.history['val_binary_accuracy'] for mh in loss_history]),
                 'r-')
    ax3.legend(['Training', 'Validation'])
    ax3.set_title('Binary Accuracy (%)')

    _ = ax4.plot(epich, np.concatenate(
        [mh.history['dice_coef'] for mh in loss_history]), 'b-',
                 epich, np.concatenate(
            [mh.history['val_dice_coef'] for mh in loss_history]),
                 'r-')
    ax4.legend(['Training', 'Validation'])
    ax4.set_title('DICE')


show_loss(loss_history)
# %%
seg_model.load_weights(weight_path)
seg_model.save('seg_model.h5')
# %%
pred_y = seg_model.predict(valid_x)
print(pred_y.shape, pred_y.min(), pred_y.max(), pred_y.mean())
# %% md
## Prepare Full Resolution Model
# %%
# if IMG_SCALING is not None:
#     fullres_model = models.Sequential()
#     fullres_model.add(layers.AvgPool2D(IMG_SCALING, input_shape = (None, None, 3)))
#     fullres_model.add(seg_model)
#     fullres_model.add(layers.UpSampling2D(IMG_SCALING))
# else:
#     fullres_model = seg_model
# fullres_model.save('fullres_model.h5')
# %% md
## Run the test data
# %%
test_paths = os.listdir(test_image_dir)
print(len(test_paths), 'test images found')
# %%
fig, m_axs = plt.subplots(20, 2, figsize=(10, 40))
[c_ax.axis('off') for c_ax in m_axs.flatten()]
for (ax1, ax2), c_img_name in zip(m_axs, test_paths):
    c_path = os.path.join(test_image_dir, c_img_name)
    c_img = imread(c_path)
    c_img = cv2.resize(c_img, (256, 256))
    first_img = np.expand_dims(c_img, 0) / 255.0
    first_seg = seg_model.predict(first_img)
    first_img[0][:, :, 0] = (first_img[0][:, :, 0] * 0.7 + 0.5 * first_seg[0, :, :, 0])
    result = np.array(np.clip(first_img[0] * 255., 0, 255), dtype=np.int32)
    ax1.imshow(result)
    ax1.set_title('Image')
    ax2.imshow(first_seg[0, :, :, 0], vmin=0, vmax=1, cmap='gray')
    ax2.set_title('Prediction')
fig.savefig('test_predictions.png')
# %%
from moviepy.editor import VideoFileClip


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)
    image_shape = image.shape[:2]
    #     print(image_shape)
    image = cv2.resize(image, (256, 256))
    first_img = np.expand_dims(image, 0) / 255.0
    #     result = image_pipeline(image)
    first_seg = seg_model.predict(first_img)
    first_img[0][:, :, 0] = first_img[0][:, :, 0] * 0.7 + 0.3 * first_seg[0, :, :, 0]
    result = np.array(np.clip(first_img[0] * 255, 0, 255), dtype=np.float)
    #     print(image_shape[:2],result.shape,type(result[0][0][0]))
    result = cv2.resize(result, image_shape[::-1])
    #     result = result[...,::-1]

    return result

# %%
filename = 'videoplayback.mp4'
clip = VideoFileClip(filename)
white_clip = clip.fl_image(process_image)
# time
white_clip.write_videofile(filename.split('.')[0] + '_detection.mp4', audio=False)


# %%
def get_video(src):
    def pred_gen(src):
        ret = True
        while ret:
            ret, image = cv2.videoCapture(src)
            image = cv2.resize(image, (256, 256))
            yield image

    return pred_gen


def make_video():
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('videoplayback_out.mp4', fourcc, 25.0, (360, 640))
    video_gen = get_video('videoplayback.mp4')
    resultant_clips = seg_model.predict_generator(video_gen)
    print('Prediction complete')
    for frame in resultant_clips:
        out.write(frame)
    out.release()
    print('Video written')


make_video()
# %%
# %%


# %%
filename = 'videoplayback.mp4'
clip = VideoFileClip(filename)
white_clip = clip.fl_image(process_image)
# time
white_clip.write_videofile(filename.split('.')[0] + '_detection.mp4', audio=False)

# %%
def get_video(src):
    def pred_gen(src):
        ret = True
        while ret:
            ret, image = cv2.videoCapture(src)
            image = cv2.resize(image, (256, 256))
            yield image

    return pred_gen

def make_video():
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter('videoplayback_out.mp4', fourcc, 25.0, (360, 640))
    video_gen = get_video('videoplayback.mp4')
    resultant_clips = seg_model.predict_generator(video_gen)
    print('Prediction complete')
    for frame in resultant_clips:
        out.write(frame)
    out.release()
    print('Video written')


