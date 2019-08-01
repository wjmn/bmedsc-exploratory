import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import time
import glob
import PIL
import pickle
from datetime import datetime
from tensorflow.keras import layers, Sequential, Model
from tensorflow.keras.layers import *
from scipy.ndimage import gaussian_filter
from grid import PolarGrid, CartesianGrid, render, render_tensor

################################################################################
# GRID
################################################################################

gridType = PolarGrid
gridParam1 = 10
gridParam2 = 10
renderSize1 = 48
renderSize2 = 48
half=True
random=True

gridVectorLength = gridParam1 * gridParam2

grid = gridType(gridParam1, gridParam2, renderSize1, renderSize2, half=half, random=random)

################################################################################
# MNIST
################################################################################

# Load MNIST dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
num_real, original_y, original_x = train_images.shape

# Define preprocessing of MNIST images
def process_images(images):
    # Upscale only to 80% of the render, to leave some padding around the digit
    up_x, up_y = int(renderSize1 * (3/4)), int(renderSize2 * (3/4))
    pad_x, pad_y = ((renderSize2 - up_x) // 2, (renderSize1 - up_y) // 2)
    pad_widths = ((pad_y, pad_y), (pad_x, pad_x))
    processed = np.array([
        np.roll(
            # Pad and resize up to the render dimensions
            np.pad(
                cv2.resize(
                    image,
                    dsize=(up_y, up_x)
                ),
                pad_widths,
                'constant',
                constant_values=0
            ),
            # Shift along the X-axis to the right
            shift=(renderSize2 // 5),
            axis=1,
        )
        for image in images
    ])
    # Scale the values from 0-255 to -1-1
    processed = (processed - 127.5) / 127.5
    return processed

# Preprocess MNIST dataset
train_images = process_images(train_images)

################################################################################
# HOLE FILLER
################################################################################

# Loading the hole-filler
filler_path = "../03-psychophysics/data/training-hole-filling/training-encoders/2019-07-18_13-57_encoder_48_48.h5"
filler = tf.keras.models.load_model(filler_path)

################################################################################
# DEFINING MODEL
################################################################################

# Defining an encoder
SEED_SIZE = 32
def make_encoder():

    # Input noise
    in_noise        = Input(shape=(SEED_SIZE,))

    # Layers for noise
    l_noise         = BatchNormalization()(in_noise)
    l_noise         = LeakyReLU()(l_noise)

    # Input labels
    in_labels       = Input(shape=(1,))

    # Layers for labels
    l_labels        = Embedding(10, gridVectorLength // 4)(in_labels)
    l_labels        = Flatten()(l_labels)
    l_labels        = BatchNormalization()(l_labels)
    l_labels        = LeakyReLU()(l_labels)

    # Concatenate
    concat          = Concatenate()([l_noise, l_labels])
    concat          = Flatten()(concat)
    concat          = Dense(gridVectorLength // 2)(concat)
    concat          = LeakyReLU()(concat)
    concat          = Concatenate()([concat, l_labels])
    concat          = Dense(gridVectorLength // 2)(concat)
    concat          = LeakyReLU()(concat)
    concat          = Concatenate()([concat, l_labels])
    concat          = Dense(gridVectorLength, activation=tf.nn.sigmoid)(concat)

    # Model
    model           = Model([in_noise, in_labels], concat)

    return model

# Defining a decoder
def make_decoder():

    # Image input
    in_images     = Input(shape=(renderSize1, renderSize2))
    l_images      = Reshape((renderSize1, renderSize2, 1))(in_images)

    # Convolution
    concat        = Conv2D(64, (8,8), padding='same', strides=(2,2))(l_images)
    concat        = LeakyReLU()(concat)
    concat        = Dropout(0.25)(concat)

    label_out     = Conv2D(128, (4,4), padding='same', strides=(2,2))(concat)
    label_out     = LeakyReLU()(label_out)
    label_out     = Dropout(0.25)(label_out)
    label_out     = Flatten()(label_out)
    label_out     = Dense(11, activation=tf.nn.softmax)(label_out)

    # Model
    model         = Model(inputs=[in_images], outputs=label_out)

    return model

# Convenience functions
@tf.function
def make_inputs(labels):

    batch_size = len(labels)

    noise = tf.random.uniform((batch_size, SEED_SIZE))
    one_hot = tf.one_hot(labels, depth=10, dtype=tf.float32)

    return (noise, one_hot)

# Loss functions
binary_cross_entropy = tf.keras.losses.BinaryCrossentropy()
categorical_cross_entropy = tf.keras.losses.CategoricalCrossentropy()

def encoder_loss(real_labels, decoded_one_hot, renders):

    real_one_hot = tf.one_hot(real_labels, depth=11)
    label_loss = categorical_cross_entropy(real_one_hot, decoded_one_hot)

    brightness_loss = tf.reduce_sum(tf.math.cos(tf.constant(np.pi / 2) * renders))

    return label_loss + (brightness_loss / (renderSize1 * renderSize2 * 100))

def decoder_loss(real_labels, decoded_real_one_hot, decoded_fake_one_hot):

    shape = real_labels.shape

    real_one_hot = tf.one_hot(real_labels, depth=11)
    real_loss = categorical_cross_entropy(real_one_hot, decoded_real_one_hot)

    all_fake_one_hot = tf.one_hot(tf.fill(shape, 10), depth=11)
    fake_loss = categorical_cross_entropy(all_fake_one_hot, decoded_fake_one_hot)

    loss = real_loss + fake_loss

    return loss

# Optimizers
encoder_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
decoder_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# Creating an encoder and decoder
encoder = make_encoder()
decoder = make_decoder()

################################################################################
# DATA STORAGE
################################################################################

save_dir = "./data/training-intermediate-data/"
base     = "{dir}/{time}_{type}_{gridType}_{gridParam1}-{gridParam2}_{renderSize1}-{renderSize2}-{half}-{random}.{ext}"
now      = datetime.now().strftime('%Y-%m-%d_%H-%M')

common_format = {
    'time': now,
    'gridType': gridType.__name__,
    'gridParam1': gridParam1,
    'gridParam2': gridParam2,
    'renderSize1': renderSize1,
    'renderSize2': renderSize2,
    'half': 'half' if half else 'full',
    'random': 'random' if random else 'regular',
}

loss_filepath = save_dir + base.format(
    dir="training-losses",
    type='loss',
    ext='log',
    **common_format
)

gif_filepath = save_dir + base.format(
    dir="training-gifs",
    type='evolution',
    ext='gif',
    **common_format
)

grid_filepath = save_dir + base.format(
    dir='training-grids',
    type='grid',
    ext='pkl',
    **common_format
)

encoder_filepath = save_dir + base.format(
    dir='training-encoders',
    type='encoder',
    ext='h5',
    **common_format
)

image_filepath = save_dir + base.format(
    dir='training-images',
    type='image',
    ext='png',
    **common_format
)

# Define the encoder and decoder log paths
enc_logpath = loss_filepath.replace("_loss_", "_enc_loss_cgan_")
dec_logpath = loss_filepath.replace("_loss_", "_dec_loss_cgan_")

# Format for tensorflow's print function
enc_logfile = "file://" + enc_logpath
dec_logfile = "file://" + dec_logpath

################################################################################
# Training functions
################################################################################

curried_render = lambda x: render_tensor(grid, x)

@tf.function
def train_step(real_images, real_labels):

    with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape:

        # Generate the encoder inputs with a random seed
        seeds, one_hot_labels = make_inputs(real_labels)

        # Produce encodings from the encoder inputs
        encodings = encoder((seeds, real_labels), training=True)

        # Render the encodings
        encoded_renders = tf.map_fn(curried_render, encodings)

        # Fill holes in encoded renders
        filled_renders = filler(encoded_renders)

        # Decode the rendered images
        decoded_fake_one_hot = decoder((filled_renders,))

        # Decode the real images
        decoded_real_one_hot = decoder((real_images,))

        # Calculate the losses for the encoder and decoder
        enc_loss = encoder_loss(real_labels, decoded_fake_one_hot, encoded_renders)
        dec_loss = decoder_loss(real_labels, decoded_real_one_hot, decoded_fake_one_hot)

    # Output the losses to the log files
    tf.print(enc_loss, output_stream=enc_logfile)
    tf.print(dec_loss, output_stream=dec_logfile)

    # Calculate and apply the gradients to the encoder and decoder
    gradients_of_encoder = enc_tape.gradient(enc_loss, encoder.trainable_variables)
    gradients_of_decoder = dec_tape.gradient(dec_loss, decoder.trainable_variables)

    encoder_optimizer.apply_gradients(zip(gradients_of_encoder, encoder.trainable_variables))
    decoder_optimizer.apply_gradients(zip(gradients_of_decoder, decoder.trainable_variables))

################################################################################
# Training display
################################################################################

display_labels = tf.convert_to_tensor(range(10))
display_seeds, display_one_hot = make_inputs(display_labels)

def generate_and_save_images(encoder, epoch):

    encodings = encoder((display_seeds, display_labels), training=False)

    fig, axes = plt.subplots(1, 10, figsize=(10, 1))

    for i in range(10):
        axes[i].imshow(curried_render(encodings[i].numpy()), cmap='gray', vmin=-1, vmax=1)
        axes[i].set_title(i)
        axes[i].axis('off')

    plt.suptitle(f"Epoch {epoch}", y=1.3)

    plt.savefig(image_filepath.replace('.png', '_epoch-{0:02d}.png'.format(epoch)))
    plt.tight_layout()
    plt.close()

generate_and_save_images(encoder, 0)

################################################################################
# First training step
################################################################################

BATCH_SIZE = 250
EPOCH_SIZE = num_real // 2
NUM_BATCHES = EPOCH_SIZE // BATCH_SIZE

# Initial step
initial_labels = train_labels[:BATCH_SIZE]
initial_real = tf.cast(train_images[:BATCH_SIZE], tf.float32)

initial_seeds, initial_one_hot = make_inputs(initial_labels)
initial_encodings = encoder((initial_seeds, initial_labels))
initial_renders = tf.map_fn(lambda x: curried_render(x.numpy()), (initial_encodings))
initial_filled = filler(initial_renders)

initial_decoded_fake = decoder((initial_filled,))
initial_decoded_real = decoder((initial_real,))

with open(enc_logpath, 'w') as outfile:
    encoder_loss_value = encoder_loss(initial_labels, initial_decoded_fake, initial_renders)
    outfile.write(str(encoder_loss_value.numpy()))
    outfile.write('\n')

with open(dec_logpath, 'w') as outfile:
    decoder_loss_value = decoder_loss(initial_labels, initial_decoded_real, initial_decoded_fake)
    outfile.write(str(decoder_loss_value.numpy()))
    outfile.write('\n')


################################################################################
# Defining the training loop
################################################################################

def train(epochs):

    for epoch in range(epochs):
        start = time.time()
        print(f'Starting epoch {epoch+1} at {time.asctime()}')

        for i in range(NUM_BATCHES):
            imin = i * BATCH_SIZE
            imax = (i+1) * BATCH_SIZE

            real_images_slice = tf.cast(train_images[imin:imax], tf.float32)
            real_labels_slice = tf.cast(train_labels[imin:imax], tf.int32)

            train_step(real_images_slice, real_labels_slice)

        generate_and_save_images(encoder, epoch + 1)

        print(f'Time for epoch {epoch+1} is {time.time()-start} sec.')

################################################################################
# Train
################################################################################

EPOCHS = 50

train(EPOCHS)

################################################################################
# Train
################################################################################

# Generate GIF
generated_images = glob.glob(image_filepath.replace(".png", "*.png"))
generated_images.sort()
images = [PIL.Image.open(image) for image in generated_images]

# save the first image 10 times
images[0].save(gif_filepath,
               save_all=True,
               append_images=[images[0]] * 10 + images + [images[-1]]*10,
               duration=100,
               loop=0)

# Save Grid
with open(grid_filepath, 'wb') as outfile:
    pickle.dump(grid, outfile)

# Save the trained encoder
encoder.save(encoder_filepath)
encoder = tf.keras.models.load_model(encoder_filepath)
