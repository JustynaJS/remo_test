import time

import tensorflow as tf
from IPython.core.display import clear_output


# @tf.function()
def train_step(input_image, target, generator, discriminator,
               generator_loss, discriminator_loss,
               generator_optimizer, discriminator_optimizer, loss_object):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_loss = generator_loss(disc_generated_output, gen_output, target, loss_object)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output, loss_object)

    generator_gradients = gen_tape.gradient(gen_loss,
                                            generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss,
                                                 discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients,
                                            generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                                discriminator.trainable_variables))


def train(dataset, epochs, generator, discriminator,
          generator_loss, discriminator_loss,
          generator_optimizer, discriminator_optimizer,
          loss_object,
          test_dataset, generate_images,
          checkpoint, checkpoint_prefix):
    for epoch in range(epochs):
        print('epoch #{}/{}'.format(epoch, epochs))
        start = time.time()

        for input_image, target in dataset:
            train_step(input_image, target, generator, discriminator, generator_loss, discriminator_loss,
                       generator_optimizer, discriminator_optimizer, loss_object)

        clear_output(wait=True)
        for inp, tar in test_dataset.take(1):
            generate_images(generator, inp, tar)

        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 20 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                           time.time() - start))
