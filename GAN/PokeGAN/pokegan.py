#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 15:15:15 2020

@author: venturus
"""
import sys
sys.path.append('..')

import pickle
import os 

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from tensorflow.keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization
from tensorflow.keras.layers import Reshape, Conv2DTranspose, LeakyReLU, Dropout, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam

from tqdm import tqdm
from IPython import display

dim = 100

def build_generator():
  tf.random.set_seed(0)
  i = Input(shape = (dim, ))
  
  x = Dense(15*15*256)(i)
  x = BatchNormalization()(x)
  x = LeakyReLU()(x)
  x = Reshape((15, 15, 256))(x)
  #(15, 15, 256)

  x = Conv2D(128, (5,5), strides=(1,1), padding='same')(x)
  x = BatchNormalization()(x)
  x = LeakyReLU()(x)
  #(15, 15, 128)
  x = UpSampling2D()(x)
  #(30, 30, 128)

  x = Conv2D(64, (5,5), strides=(1,1), padding='same')(x)
  x = BatchNormalization()(x)
  x = LeakyReLU()(x)
  #(30, 30, 64)
  x = UpSampling2D()(x)
  #(60, 60, 128)


  x = Conv2D(32, (5,5), strides=(1,1), padding='same')(x)
  x = BatchNormalization()(x)
  x = LeakyReLU()(x)
  #(60, 60, 64)

  x = Conv2DTranspose(3, (5,5), strides=(2,2), padding='same', activation="tanh")(x)
  #(120, 120, 3)
  
  
  model = Model(i, x)
  return model


def build_discriminator():
  i = Input(shape = (H, W, C))

  x = Conv2D(64, (5, 5), strides=(2,2), padding='same')(i)
  x = BatchNormalization()(x)
  x = LeakyReLU()(x)
  x = Dropout(0.3)(x)

  x = Conv2D(128, (5, 5), strides=(2,2), padding='same')(x)
  x = BatchNormalization()(x)
  x = LeakyReLU()(x)
  x = Dropout(0.3)(x)

  x = Flatten()(x)
  x = Dense(64, activation="tanh")(x)
  x = Dense(32, activation="tanh")(x)
  x = Dense(1, activation='sigmoid')(x)

  model = Model(i, x)
  return model


def save_and_show(model, epoch):
    if not os.path.exists('sample'):
        os.makedirs('sample')
    generated = model.sample(9, 0)
    generated -= generated.min()
    generated /= generated.max()
    fig = plt.figure(figsize=(10,10))
    for i in range(generated.shape[0]):
        plt.subplot(3, 3, i+1)
        plt.imshow(generated[i, :, :, :])
        plt.axis('off')
    plt.savefig(f'sample/image_at_epoch_{epoch:04d}.png')
    
    

def training(model, epochs, batch_size, data):
    pbar = tqdm(range(epochs + 1), position=0, leave=True)
    r = {'gen_loss' : [], 'disc_loss' : []}
    for epoch in pbar:
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_data = data[idx]
        gen, disc, _, _ = model.train_step(real_data)
        r['gen_loss'].append(gen)
        r['disc_loss'].append(disc)
        description = f'gen_loss {gen:.4f} disc_loss {disc:.4f}'
        pbar.set_description(description)
        
        if epoch % 200 == 0:
            save_and_show(model, epoch)
        
        if epoch % 1000 == 0:
            model.save()
    
    save_and_show(model, epoch)    
        
    return r


with open('pokemon.pkl','rb') as f:
    X = pickle.load(f)

dataset = pd.read_csv('pokemon.csv')
img_format = dataset['Name'].apply(lambda x : x.split('.')[1])
X = X[dataset.loc[(img_format == 'png')].index]
X = X / 255. * 2 -1

N, H, W, C = X.shape

from gan_models import GAN

gen_optimizer = Adam(2e-4, 0.5)
disc_optimizer = SGD()
generator = build_generator()
discriminator = build_discriminator()

gan = GAN(generator, discriminator, dim, gen_optimizer, disc_optimizer)

r = training(gan, 100000, 30, X)

plt.plot(r['disc_loss'])
plt.plot(r['gen_loss'])
plt.show()
