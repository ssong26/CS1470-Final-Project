#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import os
from matplotlib import pyplot as plt
import datetime
size = 50
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models
#import matplotlib.pyplot as plt
tf.keras.backend.set_floatx('float32')
#print("Num GPUs Availabel: ", len(tf.config.experimental.list_physical_devices("GPU")))


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

I = generate_pattern(700 * 10**(-6))
screen_length = 8
r = np.linspace(0,screen_length,size);
theta = np.linspace(0,2*np.pi,size);
r_matrix,theta_matrix = np.meshgrid(r,theta)
screen_x = r_matrix * np.cos(theta_matrix);
screen_y = r_matrix * np.sin(theta_matrix);
plt.figure(figsize=(8,8),dpi = 300)
plt.pcolormesh(screen_x,screen_y,I, cmap='gray')
plt.xlabel("x (mm)",fontsize=18)

plt.tick_params(axis='x', labelsize=18)
plt.ylabel("y (mm)",fontsize=18)

plt.tick_params(axis='y', labelsize=18)
plt.savefig("inference_pattern_700.jpg")

lam_min = 300 * 10**(-6); # wavelength of the laser
lam_max = 700 * 10**(-6); # wavelength of the laser
training_data = []
training_label = []
for i in range(0,10000):
    lam = np.random.random() * (lam_max - lam_min) + lam_min;
    I = generate_pattern(lam)
    training_data.append(I.reshape([size,size,1]))
    training_label.append([(lam - lam_min)/(lam_max - lam_min)])
training_data = np.array(training_data)
training_label = np.array(training_label)
#
lam_min = 300 * 10**(-6); # wavelength of the laser
lam_max = 700 * 10**(-6); # wavelength of the laser
testing_data = []
testing_label = []
for i in range(0,100):
    lam = np.random.random() * (lam_max - lam_min) + lam_min;
    I = generate_pattern(lam)
    testing_data.append(I.reshape([size,size,1]))
    testing_label.append([(lam - lam_min)/(lam_max - lam_min)])
testing_data = np.array(training_data)
testing_label = np.array(training_label)

model = models.Sequential()
model.add(layers.Conv2D(8,(3,3),padding = "SAME",use_bias=True, input_shape=(size,size,1)))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(32,(3,3),padding = "SAME",use_bias=True, input_shape=(size,size,1)))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
#model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(2,(3,3),padding = "SAME",use_bias=True, input_shape=(size,size,1)))
model.add(layers.BatchNormalization())
model.add(layers.ReLU())
model.add(layers.MaxPooling2D((2, 2)))
#model.add(layers.Conv2D(4,(3,3),padding = "SAME",use_bias=True, input_shape=(size,size,1)))
#model.add(layers.BatchNormalization())
#model.add(layers.ReLU())
model.add(layers.Flatten())
#model.add(layers.Dense(30,activation='tanh'))
#model.add(layers.Dropout(0.3))
model.add(layers.Dense(8,activation='tanh'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(1))
#
model.compile(optimizer='adam',
              loss=tf.keras.losses.mean_squared_error)
#
model.summary()
#
#log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#get_ipython().system('mkdir log_dir')
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

history = model.fit(training_data, training_label, batch_size=1000,epochs=1000, validation_data=(testing_data, testing_label))

#get_ipython().run_line_magic('load_ext', 'tensorboard')

#get_ipython().run_line_magic('tensorboard', '--logdir logs/fit')

plt.figure(figsize = (8,6),dpi=300)
plt.plot(np.log10(history.history['val_loss']))
plt.xlabel("Epochs",fontsize=18)
plt.ylabel("Mean Squared Loss (Log10)",fontsize=18)
plt.tick_params(axis='x', labelsize=18)
plt.tick_params(axis='y', labelsize=18)
plt.savefig("cnn_history.jpg")

#get_ipython().system('mkdir -p saved_model_cnn')

model.save("saved_model_cnn/my_model")
