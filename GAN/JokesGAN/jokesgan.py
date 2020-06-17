#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 12:07:16 2020

@author: venturus
"""
import os
import sys
sys.path.append('..')

import pandas as pd
import numpy as np
import unicodedata
import re

import tensorflow as tf
from tensorflow.keras.layers import LSTM, Input, Dense, GlobalMaxPool1D
from tensorflow.keras.layers import BatchNormalization, LeakyReLU
from tensorflow.keras.layers import Conv1D, Conv1DTranspose, UpSampling1D
from tensorflow.keras.layers import Reshape, Lambda, Bidirectional, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from tqdm import tqdm


def build_generator():
    
    def gumbel_softmax(out, t):
        U = tf.random.uniform(tf.shape(out), minval=0, maxval=1)
        gumbel = -tf.math.log(-tf.math.log(U + 1e-20) + 1e-20)
    
        return tf.nn.softmax((1/t)*(out + gumbel))
    
    def gen():
        ## Encoder ##
        i = Input(shape=(DIM,))
        x = Reshape((DIM, 1))(i)
        
        h = Conv1D(32, 3, activation = 'relu', padding="same")(x)
        h = UpSampling1D(size = 2)(h)
        h = Conv1D(64, 5, activation = 'relu', padding="same")(h)
        h = UpSampling1D(size = 1)(h)
        h = Conv1D(128, 7, activation = 'relu', padding="same")(h)
        h = UpSampling1D(size = 1)(h)
        h = Conv1DTranspose(256, 7, activation = 'relu', padding="same")(h)
        h = GlobalMaxPool1D()(h)
        h = Dense(LATENT_DIM)(h)
        
        c = Conv1D(32, 3, activation = 'relu', padding="same")(x)
        c = UpSampling1D(size = 2)(c)
        c = Conv1D(64, 5, activation = 'relu', padding="same")(c)
        c = UpSampling1D(size = 1)(c)
        c = Conv1D(128, 7, activation = 'relu', padding="same")(c)
        c = UpSampling1D(size = 1)(c)
        c = Conv1DTranspose(256, 7, activation = 'relu', padding="same")(c)
        c = GlobalMaxPool1D()(c)
        c = Dense(LATENT_DIM)(c)
        state = [h,c]
        
        
        ## Decoder ##
        decoder_input = Input(shape=(1, V))
        decoder_lstm = LSTM(LATENT_DIM, return_sequences = True, return_state = True)
        decoder_out = Dense(V)
        
        all_outputs = []
        inputs = decoder_input
        for _ in range(MAX_SEQ_GEN):
            o, h, c = decoder_lstm(inputs, initial_state= state)
            o = decoder_out(o)
            o = gumbel_softmax(o, 0.05)
            
            all_outputs.append(o)
            inputs = o
            state = [h,c]
        
        x = Lambda(lambda x :  K.concatenate(x, axis=1))(all_outputs)
        
        model = Model([i,decoder_input],  x)
        return model
    
    
    model = gen()
    i = Input(shape = (DIM,))
    initial = tf.zeros((tf.shape(i)[0], 1, V))
    x = model([i, initial]) 
    model = Model(i, x)
    return model

def build_discriminator():
    i = Input(shape=(None, V), batch_size = BATCH_SIZE)
    x = Bidirectional(LSTM(64, return_sequences=True,
             stateful = True,
             recurrent_initializer='glorot_uniform'))(i)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Bidirectional(LSTM(32))(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    return Model(i, x)

def save_and_show(model, epoch):
    
    if not os.path.exists('sample'):
        os.makedirs('sample')
    
    generated = model.sample(9, 0).argmax(-1)
    with open('sample/sample.txt', 'a') as f:
        f.write(f"joke_at_epoch_{epoch:04d}:\n")
        for i in range(9):
            joke = "".join([idx2word[idx] for idx in generated[i]])
            f.write(f"Joke - {i:d}: {joke:s}\n\n")
        f.write(f"\n\n") 

def training(model, epochs, batch_size, data):
    pbar = tqdm(range(epochs + 1), position=0, leave=True)
    r = {'gen_loss' : [], 'disc_loss' : [], 'gen_acc' : [], 'disc_acc' : []}
    for epoch in pbar:
        idx = np.random.randint(0, data.shape[0], batch_size)
        real_data = tf.one_hot(data[idx], V)
        gen_loss, disc_loss, gen_acc, disc_acc = model.train_step(real_data)
        r['gen_loss'].append(gen_loss)
        r['disc_loss'].append(disc_loss)
        r['gen_acc'].append(gen_acc)
        r['disc_acc'].append(disc_acc)
        loss = f'gen_loss: {gen_loss:.4f} disc_loss: {disc_loss:.4f} '
        acc = f'gen_acc: {gen_acc:.4f} disc_acc: {disc_acc:.4f}'
        description = loss + acc
        pbar.set_description(description)
        
        if epoch % 200 == 0:
            save_and_show(model, epoch)
        
        if epoch % 1000 == 0:
            model.save()
            
    save_and_show(model, epoch)    
        
    return r

def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')


def create_sequence(seq):
    texts = seq
    texts = [[l for l in phrase] for phrase in texts]
    texts = [item for sublist in texts for item in sublist] 
    texts += ['']

    vocab = sorted(set(texts))
    word2idx = {u:i for i, u in enumerate(vocab)}
    idx2word = np.array(vocab)
    
    sequences = [[word2idx[w] for w in phrase] for phrase in seq]
    
    return sequences, word2idx, idx2word

df = pd.read_csv('shortjokes.csv')
df.index = df.pop('ID')

texts = df['Joke'].tolist()

## Remove jokes that need a link as the context ###
texts = [w for w in texts if not re.search(r'https?://[A-Za-z0-9./]+', w)] 
texts = [w for w in texts if not re.search(r' www.[^ ]+', w)]
texts = [w for w in texts if not re.search(r' @[^ ]+', w)]
texts = [w for w in texts if not re.search(r' #[^ ]+', w)]
                                           
#Preprocessing text
texts = [re.sub(r"([?.!,¿])", r" \1 ", w) for w in texts]
texts = [unicode_to_ascii(w.lower().strip()) for w in texts]
texts = [re.sub(r"[^a-z?.!,¿]+", " ", w) for w in texts]
texts = [re.sub(r"([a-z])\1{1,}", r"\1\1", w) for w in texts]

texts = [w.strip() for w in texts]
                                  
sequences, word2idx, idx2word = create_sequence(texts)    
del df

MAX_LENGTH = max([len([w for w in words])for words in sequences])
sequences = pad_sequences(sequences, maxlen=MAX_LENGTH, padding='post')

V = min(100000, len(word2idx))
BATCH_SIZE = 30
LATENT_DIM = 128
MAX_SEQ_GEN = 100
DIM = 100

from gan_models import GAN

gen_optimizer = Adam(2e-4, 0.5)
disc_optimizer = SGD()
generator = build_generator()
discriminator = build_discriminator()


model = GAN(generator, discriminator, DIM, gen_optimizer, disc_optimizer)

r = training(model, 100000, BATCH_SIZE, sequences)

import matplotlib.pyplot as plt

plt.plot(r['disc_loss'])
plt.plot(r['gen_loss'])
plt.show()
