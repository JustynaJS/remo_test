from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import os

# Import functions from other files
from discriminator import Discriminator, discriminator_loss
from generator import Generator, generator_loss
from read_images import load_image_train, load_image_test
from training import train
from utils import generate_images

tf.compat.v1.enable_eager_execution()

""" Pix2Pix """

# Load the dataset
# In random jittering, the image is re-sized to `286 x 286` and then randomly cropped to `256 x 256`
# In random mirroring, the image is randomly flipped horizontally i.e left to right.

# _URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'
# path_to_zip = tf.keras.utils.get_file('facades.tar.gz', origin=_URL, extract=True)
# PATH = os.path.join(os.path.dirname(path_to_zip), 'facades/')

# Load my dataset:
PATH = os.path.join('dataset', '01')

BUFFER_SIZE = 400
BATCH_SIZE = 10
# IMG_HEIGHT = 256
# IMG_WIDTH = 256
IMG_HEIGHT = 22
IMG_WIDTH = 22

""" Input Pipeline"""
train_dataset = tf.data.Dataset.list_files(os.path.join(PATH, 'train/*.png'))
train_dataset = train_dataset.map(load_image_train, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.batch(BATCH_SIZE)

test_dataset = tf.data.Dataset.list_files(os.path.join(PATH, 'test/*.png'))
# shuffling so that for every epoch a different image is generated
# to predict and display the progress of our model.
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

OUTPUT_CHANNELS = 3

# Generator
generator = Generator()

# Discriminator
discriminator = Discriminator()

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

train(train_dataset, EPOCHS, generator, discriminator,
      generator_loss, discriminator_loss,
      generator_optimizer, discriminator_optimizer,
      loss_object, test_dataset,
      checkpoint, checkpoint_prefix)

# # restoring the latest checkpoint in checkpoint_dir
# checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Run the trained model on the entire test dataset
for idx, (inp, tar) in enumerate(test_dataset.take(5)):
    generate_images(generator, inp, tar, 'final_{}'.format(idx))
