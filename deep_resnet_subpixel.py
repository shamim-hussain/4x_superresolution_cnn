# -*- coding: utf-8 -*-

from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
from data_gen import DataGen
import numpy as np
from sklearn.model_selection import train_test_split

#db_path = r'G:\mix5k.h5'
db_path = r'E:\images_uint8.h5'

train_split = .75
ishape = (16,16)
pshape = (64,64)
strides = (32,32)
bsize = 64
num_epochs=30

dg = DataGen(db_path, pshape=pshape, strides=strides, ds_method='cubic')
with dg.load_db() as db:
    all_data=dg.get_datasets()
    train,test=train_test_split(all_data, train_size=train_split, 
                                shuffle=True, random_state=7)
    train_patches = dg.get_patch_list(train)
    test_patches = dg.get_patch_list(test, pshape=pshape, strides=strides)
    print('Loading Images ....')
    dg.load_images()
    train_gen = dg.patch_gen(train_patches, bsize)
    test_gen = dg.patch_gen(test_patches, bsize)
    val_gen = dg.patch_gen(test_patches, bsize)
    len_train = len(train_patches)//bsize
    len_test = len(test_patches)//bsize
    len_train = len_train//5
    len_test = len_test//5

def prop_sigmoid(x):
    over = tf.stop_gradient(tf.maximum(x-1., 0.))
    under = tf.stop_gradient(tf.maximum(0.-x, 0.))
    return x-over+under

def subpixelate(x):
    f = 4
    b,h,w,c=tf.unstack(tf.shape(x))
    tc = x.get_shape().as_list()[-1]//(f*f)
    x = tf.reshape(x, [b,h,w,f,f,tc])
    x = tf.transpose(x, [0,5,1,3,2,4])
    x = tf.reshape(x, [b,tc,h*f,w*f])
    x = tf.transpose(x, [0,2,3,1])
    return x

K.clear_session()

in_t = Input(shape=(None,None,3), dtype='int8')
in_tx1  = Lambda(lambda v: tf.cast(v,tf.float32)/255.)(in_t)
x = in_tx1

x = Conv2D(128, (5,5), padding='same', activation=None)(x)
x = Activation('relu')(x)

y = x
x = Concatenate()([x,in_tx1])
x = Conv2D(128, (3,3), padding='same', activation=None)(x)
x = Add()([x,y])
x = Activation('relu')(x)

y = x
x = Concatenate()([x,in_tx1])
x = Conv2D(128, (3,3), padding='same',  dilation_rate=1,activation=None)(x)
x = Add()([x,y])
x = Activation('relu')(x)

y = x
y = Conv2D(256, (1,1), activation=None)(y)
x = Concatenate()([x,in_tx1])
x = Conv2D(256, (3,3), padding='same', dilation_rate=1, activation=None)(x)
x = Add()([x,y])
x = Activation('relu')(x)

y = x
x = Concatenate()([x,in_tx1])
x = Conv2D(256, (3,3), padding='same', dilation_rate=1, activation=None)(x)
x = Add()([x,y])
x = Activation('relu')(x)

y = x
y = Conv2D(512, (1,1), activation=None)(y)
x = Concatenate()([x,in_tx1])
x = Conv2D(512, (3,3), padding='same', dilation_rate=2, activation=None)(x)
x = Add()([x,y])
x = Activation('relu')(x)

y = x
x = Concatenate()([x,in_tx1])
x = Conv2D(512, (3,3), padding='same', dilation_rate=2, activation=None)(x)
x = Add()([x,y])
x = Activation('relu')(x)

x = Conv2D(3*16, (3,3), padding='same', dilation_rate=2, activation=None)(x)

x = Lambda(subpixelate)(x)
x = Activation(prop_sigmoid)(x)
out_t = x

def int_mse_loss(y_true, y_pred):
    print(y_true.dtype)
    return tf.reduce_mean((y_true/255.-y_pred)**2)


            
            
model = Model(in_t, out_t)
print(model.summary())

opt = Adam(lr=1e-4)
model.compile(opt, int_mse_loss)

def on_ep_end(epoch, log):
    plt.figure(figsize=(15,5))
    X, Y = next(val_gen)
    Y_pred = model.predict_on_batch(X)
    for k in range(15):
        c, r = k//5, k%5
        dp = r*3+c
        sp = r*11+c*4
        plt.subplot(5, 11, 1 + sp)
        plt.axis('off')
        plt.imshow(X[dp])
        if not r: plt.title('LR')
        plt.subplot(5, 11, 2 + sp)
        plt.axis('off')
        plt.imshow(np.clip(Y_pred[dp],0.,1.))
        if not r: plt.title('HRPred')
        plt.subplot(5, 11, 3 + sp)
        plt.axis('off')
        plt.imshow(Y[dp])
        if not r: plt.title('HR')

    plt.pause(.01)
    figs=plt.get_fignums()
    if len(figs)>15: 
        for ff in figs[0:len(figs)-15]: plt.close(ff)

ep_end = LambdaCallback(on_epoch_end=on_ep_end)
mchk = ModelCheckpoint('deep_resnet_subpix_checkpoint.h5',save_best_only=True)
redlr = ReduceLROnPlateau(factor=.75, patience=1, verbose=1, cooldown=1)
#model.load_weights('big_subpix_deep_resnet_incremental_v1_32x32_loss0038_linear_downsampling_imagenet_validationdata.h5')
model.fit_generator(train_gen, steps_per_epoch=len_train, epochs=num_epochs, 
                    validation_data=test_gen, validation_steps=len_test/4, 
                    callbacks=[ep_end, mchk])

model.save('big_subpix_deep_resnet_incremental_v1_32x32_loss0038_linear_downsampling_imagenet_validationdata.h5')

import cv2
img=dg.image_dict[test[5]]
imgl=cv2.resize(img, None, fx=.25, fy=.25)
imgs=model.predict_on_batch(imgl[None,...])[0]
imgr=resize(img, imgs.shape).astype('float32')

plt.figure(figsize=[7,7]), plt.axis('off')
plt.imshow(imgl)
plt.title('LO-RES')
plt.tight_layout()

plt.figure(figsize=[20,7])

plt.subplot(131),plt.axis('off')
imgc=np.clip(cv2.resize(imgl/255., None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC), 0.,1.)
plt.imshow(imgc)
plt.title('4xBI-CUBIC')

plt.subplot(132),plt.axis('off')
plt.imshow(imgs)
plt.title('4xSUPERRESOLUTION')

plt.subplot(133),plt.axis('off')
plt.imshow(imgr)
plt.title('ORIGINAL HIRES')
plt.tight_layout()


num_img = len(test)
bic_psnr_arr, bic_ssim_arr = [], [];
sup_psnr_arr, sup_ssim_arr = [], [];
for k in range(num_img):
    if not k%10:
        print('{}/{}'.format(k,num_img))
    img=dg.image_dict[test[k]]
    if img.shape[0]*img.shape[1]>400*500:
        continue
    imgl=cv2.resize(img, None, fx=.25, fy=.25)
    imgs=model.predict_on_batch(imgl[None,...])[0]
    imgr=resize(img, imgs.shape).astype('float32')
    imgc=np.clip(cv2.resize(imgl/255., None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC), 0.,1.).astype('float32')
    bic_psnr_arr.append(compare_psnr(imgr, imgc))
    sup_psnr_arr.append(compare_psnr(imgr, imgs))
    bic_ssim_arr.append(compare_ssim(imgr, imgc, multichannel=True, data_range=1.0, gaussian_weights=True))
    sup_ssim_arr.append(compare_ssim(imgr, imgs, multichannel=True, data_range=1.0, gaussian_weights=True))

sup_avg_psnr=np.mean(sup_psnr_arr)
sup_avg_ssim = np.mean(sup_ssim_arr)

bic_avg_psnr = np.mean(bic_psnr_arr)
bic_avg_ssim = np.mean(bic_ssim_arr)


print('Bi-cubic PSNR:', bic_avg_psnr)
print('Superresolution PSNR:', sup_avg_psnr)
print('Bi-cubic SSIM:', bic_avg_ssim)
print('Superresolution SSIM:', sup_avg_ssim)








