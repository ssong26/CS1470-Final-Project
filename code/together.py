#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
from matplotlib import pyplot as plt
import datetime
import time
from IPython import display

import tensorflow as tf

from tensorflow.keras import datasets, layers, models
#import matplotlib.pyplot as plt
tf.keras.backend.set_floatx('float32')

cnn_model = tf.keras.models.load_model("saved_model_cnn/my_model")

generator_model = tf.keras.models.load_model("dcgan/my_generator")

screen_length = 8
size = 50
r = np.linspace(0,screen_length,size);
theta = np.linspace(0,2*np.pi,size);
r_matrix,theta_matrix = np.meshgrid(r,theta)
screen_x = r_matrix * np.cos(theta_matrix);
screen_y = r_matrix * np.sin(theta_matrix);

seed = np.random.random([5,1])
predictions = generator_model(seed)
plt.pcolormesh(screen_x,screen_y,predictions[0,:,:,0], cmap='gray')

EPOCHS = 1000
input_dim = 1
num_image_shown = 16
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
seed = tf.random.uniform([num_image_shown,input_dim], minval=0,maxval=1, dtype=tf.dtypes.float32)

def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.pcolormesh(screen_x,screen_y,predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')
    #plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


BATCH_SIZE = 1000
input_dim = 1
mse = tf.keras.losses.mean_squared_error
@tf.function
def train_step():
    noise_input = tf.random.uniform([BATCH_SIZE, input_dim])
    with tf.GradientTape() as gen_tape:
        generated_images = generator_model(noise_input, training=True)
        cnn_label = cnn_model(generated_images)
        loss = tf.reduce_mean(mse(noise_input, cnn_label))
    gradients_of_generator = gen_tape.gradient(loss, generator_model.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_model.trainable_variables))
    return loss

def train(epochs):
    record = []
    for epoch in range(epochs):
        start = time.time()
        s = 0
        for i in range(10):
            s = s + train_step()
            
        s = s / 10
        print("current loss is: " + str(s))
        record.append(s)
        display.clear_output(wait=True)
        generate_and_save_images(generator_model,epoch + 1,seed)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    # Generate after the final epoch
    display.clear_output(wait=True)
    return record


EPOCHS = 100

record = train(EPOCHS)

plt.figure(figsize = (8,6), dpi = 300)
plt.plot(np.log10(record))
plt.xlabel("Epoch",fontsize=18)
plt.tick_params(axis='x', labelsize=18)
plt.ylabel("Mean Squared Loss",fontsize=18)

plt.tick_params(axis='y', labelsize=18)
plt.savefig("together_loss.jpg")

generated_images = generator_model(seed)


# In[50]:


cnn_label = cnn_model(generated_images)

tf.reduce_mean(mse(cnn_label, seed))

def generate_pattern(lam):
    # all the length unit is mm.
    h = 3; # distance between two mirrors.
    f = 500; # focal length of the lens
    I0 = 1; # strength of the laser. normalized.
    # 
    #screen_length = 15;
    screen_length = 8;
    r = np.linspace(0,screen_length,size);
    theta = np.linspace(0,2*np.pi,size);
    r_matrix,theta_matrix = np.meshgrid(r,theta)
    screen_x = r_matrix * np.cos(theta_matrix);
    screen_y = r_matrix * np.sin(theta_matrix);
    interference_r = np.sqrt(screen_x**2 + screen_y**2);

    angle = np.arctan(interference_r/f);
    I_delta = 2*np.pi*h/lam * np.cos(angle);

    I = 2*I0*(np.cos(I_delta))**2;
    I = I/np.max(np.max(I));
    return I


size = 50
lam = 350*10**(-6)
lam_min = 300 * 10**(-6); # wavelength of the laser
lam_max = 700 * 10**(-6); # wavelength of the laser
label = (lam - lam_min)/(lam_max - lam_min)
plt.figure(figsize = (8,8),dpi = 300)
exact_image = generate_pattern(lam);
plt.pcolormesh(screen_x,screen_y,exact_image, cmap='gray')
plt.xlabel("x (mm)",fontsize=18)
plt.tick_params(axis='x', labelsize=18)
plt.ylabel("y (mm)",fontsize=18)
plt.tick_params(axis='y', labelsize=18)
plt.savefig("correct_350.jpg")
predictions = generator_model(np.array([label]))
plt.figure(figsize = (8,8),dpi = 300)
plt.pcolormesh(screen_x,screen_y,predictions[0, :, :, 0], cmap='gray')
plt.xlabel("x (mm)",fontsize=18)
plt.tick_params(axis='x', labelsize=18)
plt.ylabel("y (mm)",fontsize=18)
plt.tick_params(axis='y', labelsize=18)
plt.savefig("fake_350.jpg")
