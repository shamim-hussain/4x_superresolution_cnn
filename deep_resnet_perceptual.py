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
from keras.applications.vgg16 import VGG16

#db_path = r'G:\mix5k.h5'
db_path = r'E:/images_uint8.h5'

train_split = .75
ishape = (16,16)
pshape = (32,32)
strides = (16,16)
bsize = 64
num_epochs=30

reload_data = False
if 'dg' not in globals() or input('Reload data? [Y/N] : ').lower()=='y':
    reload_data=True
    dg = DataGen(db_path, pshape=pshape, strides=strides)
    with dg.load_db() as db:
        all_data=dg.get_datasets()
        train,test=train_test_split(all_data, train_size=train_split, 
                                    shuffle=True, random_state=7)
        train_patches = dg.get_patch_list(train)
        test_patches = dg.get_patch_list(test, pshape=(64,64), strides=(32,32))
        print('Loading Images ....')
        dg.load_images()
        train_gen = dg.patch_gen(train_patches, bsize)
        test_gen = dg.patch_gen(test_patches, bsize)
        len_train = len(train_patches)//bsize
        len_test = len(test_patches)//bsize

def prop_sigmoid(x):
    over = tf.stop_gradient(tf.maximum(x-1., 0.))
    under = tf.stop_gradient(tf.maximum(0.-x, 0.))
    return x-over+under

K.clear_session()
vgg16 = VGG16(include_top=False, input_shape=(64,64, 3), pooling='avg')
vgg_lrs, vgg_cfg, vgg_wts = zip(*((lr.__class__, lr.get_config(), lr.get_weights()) for lr in vgg16.layers[1:]))
del vgg16
K.clear_session()

in_t = Input(shape=(None,None,3), dtype='int8')
in_tx1  = Lambda(lambda v: tf.cast(v,tf.float32)/255.)(in_t)
x = in_tx1

x = Conv2D(64, (3,3), padding='same', activation=None)(x)
x = Activation('relu')(x)

y = Conv2D(128, (1,1))(x)
x = Concatenate()([x,in_tx1])
x = Conv2D(128, (3,3), padding='same', activation=None)(x)
x = Add()([x,y])
x = Activation('relu')(x)

y = UpSampling2D((2,2))(x)
x = Concatenate()([x,in_tx1])
x = Conv2DTranspose(128, (4,4), padding='same', strides=(2,2),
                    activation=None)(x)
x = Add()([x,y])
x = Activation('relu')(x)

in_tx2= Lambda(lambda v: tf.image.resize_bicubic(v, tf.shape(v)[1:3]*2))(in_tx1)
y = x
x = Concatenate()([x,in_tx2])
x = Conv2D(128, (3,3), padding='same', activation=None)(x)
y = Add()([x,y])
x = Activation('relu')(x)

y = UpSampling2D((2,2))(x)
x = Concatenate()([x,in_tx2])
x = Conv2DTranspose(128, (4,4), padding='same', strides=(2,2),
                    activation=None)(x)
x = Add()([x,y])
x = Activation('relu')(x)

in_tx4  = Lambda(lambda v: tf.image.resize_bicubic(v, tf.shape(v)[1:3]*4))(in_tx1)

y = x
x = Concatenate()([x,in_tx4])
x = Conv2D(128, (3,3), padding='same', activation=None)(x)
y = Add()([x,y])
x = Activation('relu')(x)

y = Conv2D(64, (1,1))(x)
x = Concatenate()([x,in_tx4])
x = Conv2D(64, (3,3), padding='same', activation=None)(x)
y = Add()([x,y])
x = Activation('relu')(x)

x = Concatenate()([x,in_tx4])
x = Conv2D(3, (3,3), padding='same', activation=None)(x)
x = Add()([x, in_tx4])
x = Activation(prop_sigmoid, name='out')(x)
out_t = x


vgg_in = Input((None, None, 3))
vgg_outs = []
x = vgg_in
for k, lr, cg, wt in zip(range(8),vgg_lrs,vgg_cfg, vgg_wts):
    if 'name' in cg:
        del cg['name']
        
    mlr = lr.from_config(cg)
    x = mlr(x)
    mlr.set_weights(wt)
    mlr.trainable=False
    vgg_outs.append(x)

vgg_model = Model(vgg_in, vgg_outs)  
vgg_model.trainable = False

def tv_loss(y_true, y_pred):
    area = tf.cast(tf.reduce_prod(tf.shape(y_pred)[1:3]), tf.float32) 
    return 0.1*tf.reduce_mean(tf.image.total_variation(y_pred)/area)

def mse_loss(y_true, y_pred):
    #print(y_true.dtype)
    return tf.reduce_mean((y_true/255.-y_pred)**2)

def percep_loss(y_true, y_pred):
    #print(y_true.dtype)
    assert y_true.dtype == tf.float32
    y = y_true/255.
    x = y_pred
    yo = vgg_model(y)
    xo = vgg_model(x)
    return (#0.01*tf.reduce_mean((yo[4]-xo[4])**2)
#            +0.01*tf.reduce_mean((yo[3]-xo[3])**2)
            +0.01*tf.reduce_mean((yo[7]-xo[7])**2)
#            +0.001*tf.reduce_mean((yo[0]-xo[0])**2)
#            +tf.reduce_mean((y-x)**2)
             ) + tv_loss(y_true,y_pred)

            
            
model = Model(in_t, out_t)
print(model.summary())

  

opt = Adam(lr=1e-4)
model.compile(opt, percep_loss, metrics=[mse_loss, tv_loss])

def on_ep_end(epoch, log):
    plt.figure(figsize=(15,5))
    X, Y = next(test_gen)
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
mchk = ModelCheckpoint('deep_resnet_checkpoint.h5',save_best_only=True)
redlr = ReduceLROnPlateau(factor=.75, patience=3, verbose=2, cooldown=2, min_lr=1e-5)

#model.load_weights('htv_training_highly_perceptual_deep_resnet_incremental_v1_32x32_linear_downsampling_mix5k_validationdata.h5')
model.fit_generator(train_gen, steps_per_epoch=len_train//10, epochs=num_epochs, 
                    validation_data=test_gen, validation_steps=len_test//10, 
                    callbacks=[ep_end, mchk])

model.save('htv_training_highly_perceptual_deep_resnet_incremental_v1_32x32_linear_downsampling_mix5k_validationdata.h5')

#import cv2
#img=dg.image_dict[test[3]]
#imgl=cv2.resize(img, None, fx=.25, fy=.25)
#imgs=model.predict_on_batch(imgl[None,...])[0]
#
#plt.figure(figsize=[7,7]), plt.axis('off')
#plt.imshow(imgl)
#plt.title('LO-RES')
#plt.tight_layout()
#
#plt.figure(figsize=[20,7])
#
#plt.subplot(131),plt.axis('off')
#plt.imshow(cv2.resize(imgl, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC))
#plt.title('4xBI-CUBIC')
#
#plt.subplot(132),plt.axis('off')
#plt.imshow(imgs)
#plt.title('4xSUPERRESOLUTION')
#
#plt.subplot(133),plt.axis('off')
#plt.imshow(img)
#plt.title('ORIGINAL HIRES')
#plt.tight_layout()
#
#


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







