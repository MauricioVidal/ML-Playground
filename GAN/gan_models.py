#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 21:13:09 2020

@author: venturus
"""
import os

import tensorflow as tf
import numpy as np

class GAN:
    
    def __discriminator_acc(self, real_output, fake_output):
        ones = tf.ones_like(real_output)
        zeros = tf.zeros_like(fake_output)
        
        acc_real = tf.metrics.binary_accuracy(ones, real_output)
        acc_fake = tf.metrics.binary_accuracy(zeros, fake_output)
        
        return tf.reduce_mean((acc_real + acc_fake)/2.)
    
    def __generator_acc(self, fake_output):
        ones = tf.ones_like(fake_output)
        return tf.reduce_mean(tf.metrics.binary_accuracy(ones, fake_output))
    
    
    def __generator_loss(self, fake_output):
        ones = tf.ones_like(fake_output)
        return tf.reduce_mean(tf.losses.binary_crossentropy(ones, fake_output))
    
    def __discriminator_loss(self, real_output, fake_output):
        range_one = tf.constant(1.2 - 0.7)
        gap_one = tf.constant(0.7)
        ones = tf.random.normal(real_output.shape)
        ones = ones - tf.math.reduce_min(ones)
        ones = (ones / (tf.math.reduce_max(ones)+0.0001)) * range_one + gap_one
        real_loss = tf.losses.binary_crossentropy(ones, real_output)
        
        range_zero = tf.constant(0.3 - 0)
        gap_zero = tf.constant(0.)
        zeros = tf.random.normal(fake_output.shape)
        zeros = zeros - tf.math.reduce_min(zeros)
        zeros = (zeros / (tf.math.reduce_max(zeros)+0.0001)) * range_zero + gap_zero
        fake_loss = tf.losses.binary_crossentropy(zeros, fake_output)
        
        return tf.reduce_mean(real_loss + fake_loss)
        
    
    def __init__(self, generator, discriminator, generator_dim,
                 gen_optimizer, disc_optimizer, 
                 random_fun = lambda shape : tf.random.normal(shape)):
        
        self.checkpoint_dir = './training_checkpoints'
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(generator_optimizer=gen_optimizer,
                                     discriminator_optimizer=disc_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)
        
        self.generator = generator
        self.discriminator = discriminator
        self.generator_dim = generator_dim
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.random_fun = random_fun
    
    def save(self):
        self.checkpoint.save(file_prefix = self.checkpoint_prefix)
        
    def restore(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
        
    def sample(self, number, seed = None):
        return self.__sample(number, False, seed).numpy()
    
    def decision(self, data):
        return self.__decision(data, False).numpy()
    
    def __sample(self, number, training, seed = None):
        tf.random.set_seed(seed)
        noise = self.random_fun((number, self.generator_dim))
        return self.generator(noise, training = training)
    
    def __decision(self, data, training):
        return self.discriminator(data, training)
    
    @tf.function
    def train_step(self, real_data):
        generator = self.generator
        discriminator = self.discriminator
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_data = self.__sample(real_data.shape[0], training=True)
            
            real_output = self.__decision(real_data, training = True)
            fake_output = self.__decision(generated_data, training = True)
            
            
            gen_loss = self.__generator_loss(fake_output)
            disc_loss =  self.__discriminator_loss(real_output, fake_output)
            
        
        grad_of_gen = gen_tape.gradient(gen_loss, generator.trainable_variables)
        grad_of_disc = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
        
        
        self.gen_optimizer.apply_gradients(zip(grad_of_gen, generator.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(grad_of_disc, discriminator.trainable_variables))
        
        disc_acc = self.__discriminator_acc(real_output, fake_output)
        gen_acc = self.__generator_acc(fake_output)
        
        
        return gen_loss, disc_loss, gen_acc, disc_acc