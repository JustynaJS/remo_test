from __future__ import absolute_import, division, print_function, unicode_literals

import time
from random import randint

import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

# Import functions from other files
import read_images
from discriminator import Discriminator, discriminator_loss
from generator import Generator, generator_loss
from read_images import load, random_jitter, load_image_train, load_image_test
from utils import downsample, upsample
from training import train

tf.compat.v1.enable_eager_execution()

# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0

""" Pix2Pix """

# Load the dataset
# In random jittering, the image is re-sized to `286 x 286` and then randomly cropped to `256 x 256`
# In random mirroring, the image is randomly flipped horizontally i.e left to right.

# _URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'
# path_to_zip = tf.keras.utils.get_file('facades.tar.gz', origin=_URL, extract=True)
# PATH = os.path.join(os.path.dirname(path_to_zip), 'facades/')

# Load my dataset:
PATH = 'R:\dataset'

# print("Loading images...")
# start = time.time()
# imgs = read_images.load_images(my_path)
# end = time.time()

BUFFER_SIZE = 400
BATCH_SIZE = 1

inp, re = load(PATH + 'train/c0.png')


# A = np.zeros((IMG_HEIGHT, IMG_WIDTH))
# for x in range(1, IMG_HEIGHT):
#     for y in range(1, IMG_WIDTH):
#         a = randint(0, 65535)
#         A[x, y] = a
#
# plt.imshow(A)
# casting to int for matplotlib to show the image
plt.figure()
plt.imshow(inp / 255.0)
plt.figure()
plt.imshow(re / 255.0)

# Pictures are going through random jittering
# Random jittering as described in the paper is to:
# 1. Resize an image to bigger height and width
# 2. Randomly crop to the original size
# 3. Randomly flip the image horizontally

# print("Jittering images...")
#
# plt.figure(figsize=(6, 6))
#
# for i in range(4):
#     rj_inp, rj_re = random_jitter(inp, re)
#     plt.subplot(2, 2, i + 1)
#     plt.imshow(rj_inp / 255.0)
#     plt.axis('off')
# plt.show()

""" Input Pipeline"""
train_dataset = tf.data.Dataset.list_files(PATH + 'train/*.jpg')
# No shuffle as samples are completely independent.
# train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(1)

test_dataset = tf.data.Dataset.list_files(PATH + 'test/*.jpg')
# shuffling so that for every epoch a different image is generated
# to predict and display the progress of our model.
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(1)

OUTPUT_CHANNELS = 3

# down_model = downsample(3, 4)
# down_result = down_model(tf.expand_dims(inp, 0))
# print(down_result.shape)
#
# up_model = upsample(3, 4)
# up_result = up_model(down_result)
# print(up_result.shape)

# Generator
generator = Generator()
gen_output = generator(inp[tf.newaxis, ...], training=False)
plt.imshow(gen_output[0, ...])

# Discriminator
discriminator = Discriminator()
disc_out = discriminator([inp[tf.newaxis, ...], gen_output], training=False)
plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
plt.colorbar()

# Define the loss functions and the optimizer
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Checkpoints: object-based saving
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

"""## Training

* We start by iterating over the dataset
* The generator gets the input image and we get a generated output.
* The discriminator receives the input_image and the generated image as the first input. 
The second input is the input_image and the target_image.
* Next, we calculate the generator and the discriminator loss.
* Then, we calculate the gradients of loss with respect to both the generator and the discriminator variables(inputs) and apply those to the optimizer.
* This entire procedure is shown in the images below.

## Generate Images

* After training, its time to generate some images!
* We pass images from the test dataset to the generator.
* The generator will then translate the input image into the output we expect.
* Last step is to plot the predictions and **voila!**
"""

EPOCHS = 150


def generate_images(model, test_input, tar):
    # the training=True is intentional here since
    # we want the batch statistics while running the model
    # on the test dataset. If we use training=False, we will get
    # the accumulated statistics learned from the training dataset
    # (which we don't want)
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15, 15))

    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        # getting the pixel values between [0, 1] to plot it.
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


train(train_dataset, EPOCHS, generator, discriminator,
      generator_loss, discriminator_loss,
      generator_optimizer, discriminator_optimizer,
      loss_object,
      test_dataset, generate_images,
      checkpoint, checkpoint_prefix)

# Restore the latest checkpoint and test
# !ls {checkpoint_dir}

# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Generate using test dataset

# Run the trained model on the entire test dataset
for inp, tar in test_dataset.take(5):
    generate_images(generator, inp, tar)
