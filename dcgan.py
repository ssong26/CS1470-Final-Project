#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import datetime
#from matplotlib import pyplot as plt
size = 50
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


from matplotlib import pyplot as plt
import time
from IPython import display
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import datetime
#import matplotlib.pyplot as plt
tf.keras.backend.set_floatx('float32')


lam_min = 300 * 10**(-6); # wavelength of the laser
lam_max = 700 * 10**(-6); # wavelength of the laser
training_data = []
training_label = []
max_num_data = 10000
for i in range(0,max_num_data):
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

# dcGAN
def create_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(5*5*20, use_bias=False, input_shape=(1,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((5,5,20)))
    #(x, 5, 5, 20)
    model.add(layers.Conv2DTranspose(20, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    #(x, 5, 5, 20)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #
    model.add(layers.Conv2DTranspose(40, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    #(x, 10, 10, 40)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    #
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(5, 5), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 50, 50, 1)

    return model

generator = create_generator_model()
generator.summary()

noise = tf.random.uniform([2,1], minval=0,maxval=1, dtype=tf.dtypes.float32)
I = generator(noise, training=False)
#
screen_length = 8
size = 50
r = np.linspace(0,screen_length,size);
theta = np.linspace(0,2*np.pi,size);
r_matrix,theta_matrix = np.meshgrid(r,theta)
screen_x = r_matrix * np.cos(theta_matrix);
screen_y = r_matrix * np.sin(theta_matrix);
plt.pcolormesh(screen_x,screen_y,I[0,:,:,0], cmap='gray')

def create_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(8,(3,3),strides = (2,2),padding = "SAME",use_bias=True, input_shape=(size,size,1)))
    #model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU ())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(32,(3,3),strides = (2,2),padding = "SAME",use_bias=True, input_shape=(size,size,1)))
    #model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU ())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    #model.add(layers.Dense(30,activation='tanh'))
    #model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1))

    return model

discriminator = create_discriminator_model()
discriminator.summary()

decision = discriminator(I)
print (decision)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

EPOCHS = 1000
input_dim = 1
num_image_shown = 16

seed = tf.random.uniform([num_image_shown,input_dim], minval=0,maxval=1, dtype=tf.dtypes.float32)

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise_input = tf.random.uniform([BATCH_SIZE, input_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise_input, training=True)
        true_output = discriminator(images, training=True)
        false_output = discriminator(generated_images, training=True)
        g_loss = generator_loss(false_output)
        d_loss = discriminator_loss(true_output, false_output)
    gradients_of_generator = gen_tape.gradient(g_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(d_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return (g_loss,d_loss)


def train(dataset, epochs):
    record = []
    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            record.append(train_step(image_batch))
        # Produce images for the GIF as you go
        display.clear_output(wait=True)
        generate_and_save_images(generator,epoch + 1,seed)
        #if (epoch + 1) % 15 == 0:
        #    checkpoint.save(file_prefix = checkpoint_prefix)
        print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,epochs,seed)
    return record

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
train_dataset = tf.data.Dataset.from_tensor_slices(training_data).shuffle(max_num_data).batch(BATCH_SIZE)

record = train(train_dataset, EPOCHS)

get_ipython().system('mkdir -p dcgan')
generator.save("dcgan/my_generator")
discriminator.save("dcgan/my_discriminator")
