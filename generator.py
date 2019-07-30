import tensorflow as tf

from utils import downsample, upsample
"""## Build the Generator
  * The architecture of generator is a modified U-Net.
  * Each block in the encoder is (Conv -> Batchnorm -> Leaky ReLU)
  * Each block in the decoder is (Transposed Conv -> Batchnorm -> Dropout(applied to the first 3 blocks) -> ReLU)
  * There are skip connections between the encoder and decoder (as in U-Net).
"""

OUTPUT_CHANNELS = 3
LAMBDA = 100



def Generator():
    # For convolutions arithmetics see: https://arxiv.org/abs/1603.07285

    down_stack = [
        downsample(64, 4, apply_batch_norm=False, strides=3, name='down_0'),  # (bs, 8, 8, 64)
        downsample(128, 4, name='down_1'),  # (bs, 4, 4, 128)
        downsample(256, 3, strides=1, name='down_2'),  # (bs, 4, 4, 256)
        downsample(512, 3, name='down_3'),  # (bs, 2, 2, 512)
        downsample(512, 4, name='down_4'),  # (bs, 1, 1, 512)
    ]


    up_stack = [
        upsample(512, 4, apply_dropout=True, name='up_0'),  # (bs, 2, 2, 512) - after concatenation (bs, 2, 2, 1024)
        upsample(256, 4, apply_dropout=True, name='up_1'),  # (bs, 4, 4, 256) - after concatenation (bs, 4, 4, 512)
        upsample(128, 4, apply_dropout=True, strides=1, name='up_2'),  # (bs, 4, 4, 128) - after concatenation (bs, 4, 4, 256)
        upsample(64, 2, apply_dropout=True, strides=2, name='up_3', padding='valid'),  # (bs, 8, 8, 64) - after concatenation (bs, 8, 8, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 8,
                                           strides=2,
                                           padding='valid',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (bs, 22, 22, 3)

    concat = tf.keras.layers.Concatenate()

    inputs = tf.keras.layers.Input(shape=[None, None, 3])
    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


"""
* **Generator loss**
  * It is a sigmoid cross entropy loss of the generated images and an **array of ones**.
  * The [paper](https://arxiv.org/abs/1611.07004) also includes L1 loss which is MAE (mean absolute error) 
  between the generated image and the target image.
  * This allows the generated image to become structurally similar to the target image.
  * The formula to calculate the total generator loss = gan_loss + LAMBDA * l1_loss, where LAMBDA = 100. 
  This value was decided by the authors of the [paper](https://arxiv.org/abs/1611.07004).
"""


def generator_loss(disc_generated_output, gen_output, target, loss_object):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

    total_gen_loss = gan_loss + (LAMBDA * l1_loss)

    return total_gen_loss
