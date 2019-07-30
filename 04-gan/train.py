import tensorflow as tf
import keras
import cv2
import numpy as np
import mnist
from datetime import datetime
from keras.layers import *
from keras import Model
from keras.utils import CustomObjectScope
from keras.initializers import glorot_uniform
from matplotlib import pyplot as plt
from grid import *


# CONFIGURING
# ------------------------------------------------------------------------------

config = {
    "grid_type": PolarGrid,
    "grid_kwargs": {"n_theta": 8, "n_radius": 8},
    "grid_vector_length": 64,
    "x_render": 48,
    "y_render": 48,
    "random": True,
    "half": True,
}

binary_cross_entropy = keras.losses.binary_crossentropy

categorical_cross_entropy = keras.losses.categorical_crossentropy

SEED_SIZE = 32

# Define the encoder
def make_encoder():

    # Input noise
    in_noise        = Input(shape=(SEED_SIZE,))
    
    # Layers for noise
    l_noise         = Dense(config["grid_vector_length"] // 4)(in_noise)
    l_noise         = BatchNormalization()(l_noise)
    l_noise         = LeakyReLU()(l_noise)
    
    # Input labels
    in_labels       = Input(shape=(1,))
    
    # Layers for labels
    l_labels        = Embedding(10, config["grid_vector_length"] // 4)(in_labels)
    l_labels        = Flatten()(l_labels)
    l_labels        = BatchNormalization()(l_labels)
    l_labels        = LeakyReLU()(l_labels)
    
    # Concatenate
    concat          = Concatenate()([l_noise, l_labels])
    concat          = Dense(config["grid_vector_length"], activation=tf.nn.sigmoid)(concat)
    
    # Model
    model           = Model([in_noise, in_labels], concat)
    
    return model

def encoder_loss(
    real_labels, 
    decoded_one_hot, 
    decoded_binary
):
    
    real_one_hot = tf.one_hot(real_labels, depth=10)
    label_loss = categorical_cross_entropy(real_one_hot, decoded_one_hot)
    
    all_real_binary = tf.ones_like(decoded_binary)
    binary_loss = binary_cross_entropy(decoded_binary, all_real_binary)
    
    return label_loss + binary_loss

encoder_optimizer = tf.keras.optimizers.Adam(lr=1e-4)

# Define the decoder
def make_decoder():
    
    # Image input
    in_images     = Input(shape=(config["y_render"], config["x_render"]))
    l_images      = Reshape((config["y_render"], config["x_render"], 1))(in_images)
    
    # Convolution
    concat        = Conv2D(64, (8,8), padding='same', strides=(2,2))(l_images)
    concat        = LeakyReLU()(concat)
    concat        = Dropout(0.25)(concat)
    
    label_out     = Conv2D(128, (4,4), padding='same', strides=(2,2))(concat)
    label_out     = LeakyReLU()(label_out)
    label_out     = Dropout(0.25)(label_out)
    label_out     = Flatten()(label_out)
    label_out     = Dense(10, activation=tf.nn.softmax)(label_out)
    
    binary_out    = Conv2D(128, (4,4), padding='same', strides=(2,2))(concat)
    binary_out    = LeakyReLU()(binary_out)
    binary_out    = Dropout(0.25)(binary_out)
    binary_out    = Flatten()(binary_out)
    binary_out    = Dense(1, activation=tf.nn.sigmoid)(binary_out)
    
    # Model 
    model         = Model(inputs=[in_images], outputs=[binary_out, label_out])
    
    return model        

def decoder_loss(
    real_labels, 
    decoded_real_one_hot, 
    decoded_real_binary, 
    decoded_fake_binary
):
    
    shape = real_labels.shape
        
    real_one_hot = tf.one_hot(real_labels, depth=10)
    label_loss = categorical_cross_entropy(real_one_hot, decoded_real_one_hot)
    
    all_real_binary = tf.ones_like(decoded_real_binary)
    real_binary_loss = binary_cross_entropy(all_real_binary, decoded_real_binary)
    
    all_fake_binary = tf.zeros_like(decoded_fake_binary)
    fake_binary_loss = binary_cross_entropy(all_fake_binary, decoded_fake_binary)
    
    loss = label_loss + real_binary_loss + fake_binary_loss
    
    return loss

decoder_optimizer = tf.keras.optimizers.Adam(lr=1e-4)

# PREPARING FOR DATA STORAGE
# ------------------------------------------------------------------------------

save_dir = "./data/training-intermediate-data/"
base     = "{dir}/{time}_{type}_{gridType}_{x_render}-{y_render}.{ext}"
now      = datetime.now().strftime('%Y-%m-%d_%H-%M')

common_format = {
    'time': now,
    'gridType': config["grid_type"].__name__,
    'x_render': config["x_render"],
    'y_render': config["y_render"],
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

# DEFINING THE STATIC GRAPH
# ------------------------------------------------------------------------------

encoder = make_encoder()

decoder = make_decoder()

filler_path = "../03-psychophysics/data/training-hole-filling/training-encoders/2019-07-15_16-47_encoder_48_48.h5"

with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
    filler = keras.models.load_model(filler_path)

train_images = mnist.load_and_preprocess(config["x_render"], config["y_render"])

def make_inputs(labels):
    
    batch_size = len(labels)
    
    noise = tf.random.uniform((batch_size, SEED_SIZE))
    one_hot = tf.one_hot(labels, depth=10, dtype=tf.float32)
    
    return (noise, one_hot)

def train_step(real_images, real_labels):
    
    with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape:
        
        seeds, one_hot_labels = make_inputs(real_labels)
        encodings = encoder((seeds, real_labels), training=True)
        encoded_renders = tf.map_fn(lambda x: render_tensor(grid, x), encodings)
        filled_renders = filler(encoded_renders)
        
        decoded_fake_binary, decoded_fake_one_hot = decoder((filled_renders,))
        decoded_real_binary, decoded_real_one_hot = decoder((real_images,))

        enc_loss = encoder_loss(real_labels, decoded_fake_one_hot, decoded_fake_binary)
        dec_loss = decoder_loss(real_labels, decoded_real_one_hot, decoded_real_binary, decoded_fake_binary)

    tf.print(enc_loss, output_stream=enc_logfile)
    tf.print(dec_loss, output_stream=dec_logfile)
    
    gradients_of_encoder = enc_tape.gradient(enc_loss, encoder.trainable_variables)
    gradients_of_decoder = dec_tape.gradient(dec_loss, decoder.trainable_variables)

    encoder_optimizer.apply_gradients(zip(gradients_of_encoder, encoder.trainable_variables))
    decoder_optimizer.apply_gradients(zip(gradients_of_decoder, decoder.trainable_variables))

def generate_and_save_images(encoder, epoch):
    
    encodings = encoder((display_seeds, display_labels), training=False)

    fig = plt.figure(figsize=(5,3))

    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(grid.render_tensor(encodings[i].numpy()),
                   cmap='gray',
                   vmin=-1,
                   vmax=1)
        plt.title(i)
        plt.axis('off')
        
    plt.suptitle(f"Epoch {epoch}")

    plt.savefig(image_directory + 'generated-epoch-{0:02d}.png'.format(epoch))
    plt.show()

# CREATING VARIABLES
# ------------------------------------------------------------------------------

grid = config["grid_type"](
    x_render=config["x_render"],
    y_render=config["y_render"],
    half=config["half"],
    random=config["random"],
    **config["grid_kwargs"]
)

# RUNNING SESSION

# Define the encoder and decoder log paths
enc_logpath = loss_filepath.replace("_loss_", "_enc_loss_cgan_")
dec_logpath = loss_filepath.replace("_loss_", "_dec_loss_cgan_")

# Format for tensorflow's print function
enc_logfile = "file://" + enc_logpath
dec_logfile = "file://" + dec_logpath

BATCH_SIZE = 250
EPOCH_SIZE = len(train_images)
NUM_BATCHES = EPOCH_SIZE // BATCH_SIZE

with open(enc_logpath, 'w') as outfile:
    pass

with open(dec_logpath, 'w') as outfile:
    pass

def train(epochs):
    
    for epoch in range(epochs):
        start = time.time()

        for i in range(NUM_BATCHES):
            imin = i * BATCH_SIZE
            imax = (i+1) * BATCH_SIZE
            
            real_images_slice = tf.cast(train_images[imin:imax], tf.float32)
            real_labels_slice = tf.cast(train_labels[imin:imax], tf.int32)
            
            train_step(real_images_slice, real_labels_slice)

        # Generate and save progressive images
        display.clear_output(wait=True)
        generate_and_save_images(encoder, epoch + 1)

        print(f'Time for epoch {epoch+1} is {time.time()-start} sec.')
        
EPOCHS = 50        

with tf.Session() as session:
    traing(EPOCHS)