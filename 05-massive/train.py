""" Runs the main training loop for a given grid, encoder_model, and decoder_model.

"""
import argparse
import os
import time
import modules.config
import modules.images
import modules.losses
import tensorflow as tf
from matplotlib import pyplot as plt
from enum import Enum

# Initialise an argument parser and parse the command-line arguments.
parser = argparse.ArgumentParser(description='Train an encoder for a given grid.')
parser.add_argument('-g', '--grid-id', help='Grid ID to use, including the G.')
parser.add_argument('-e', '--encoder-id', help='Encoder ID to use, including the E.')
parser.add_argument('-d', '--decoder-id', help='Decoder ID to use, including the D')
parser.add_argument('-f', '--filler-id', default=None, help='Filler ID to use, including the F.')
parser.add_argument('--loss-encoder', default='basic', help='Type of loss for training the encoder.')
parser.add_argument('--loss-decoder', default='basic', help='Type of loss for training the decoder.')
parser.add_argument('--learning-rate-encoder', type=float, help='Learning rate of the encoder.')
parser.add_argument('--learning-rate-decoder', type=float, help='Learning rate of the decoder')
ARGS = parser.parse_args()

# Initialise the Config object and save it.
CONFIG = modules.config.Config(base_dir='.', **vars(ARGS))
CONFIG.save()

# For convenience, take out the relevant constants from the config.
GRID = CONFIG.grid
encoder = CONFIG.encoder  # encoder will be mutated, hence lowercase!
decoder = CONFIG.decoder  # decoder will be mutated, hence lowercase!
FILLER = CONFIG.filler
LEARNING_RATE_ENCODER = CONFIG.learning_rate_encoder
LEARNING_RATE_DECODER = CONFIG.learning_rate_decoder
BATCH_SIZE = CONFIG.batch_size
NUM_EPOCHS = CONFIG.num_epochs

# Load the training data and pre-process
MNIST = modules.images.Mnist()
MNIST.preprocess_for_grid(GRID)

# Initiate the loss functions and optimizers
LOSS_ENCODER = modules.losses.DICT_LOSS_ENCODER[CONFIG.loss_encoder]
LOSS_DECODER = modules.losses.DICT_LOSS_DECODER[CONFIG.loss_decoder]
ENCODER_OPTIMISER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_ENCODER)
DECODER_OPTIMISER = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE_DECODER)


# Specify and create save directories
class Path(Enum):
    CHECKPOINT = f'./output/{CONFIG.config_id}/checkpoints/'
    ENCODER_IMAGE = f'./output/{CONFIG.config_id}/encoder_images/'
    LOSS_ENCODER = f'./output/{CONFIG.config_id}/encoder_losses.log'
    LOSS_DECODER = f'./output/{CONFIG.config_id}/decoder_losses.log'
    ENCODER_TRAINED = f'./output/{CONFIG.config_id}/encoder.h5'


LOSS_ENCODER_STREAM = 'file://' + Path.LOSS_ENCODER.value
LOSS_DECODER_STREAM = 'file://' + Path.LOSS_DECODER.value


for directory in list(Path):
    if directory.value.endswith('/'):
        os.mkdir(directory.value)


# Define training functions
NUM_BATCHES = MNIST.length // BATCH_SIZE


@tf.function
def train_step(real_images: tf.Tensor, real_labels: tf.Tensor) -> None:
    """ Trains encoder and decoder for a single batch of images and labels.
    """

    with tf.GradientTape() as enc_tape, tf.GradientTape() as dec_tape:
        encodings = encoder(real_labels, training=True)
        encoded_renders = tf.map_fn(GRID.render_values_tensor, encodings)
        filled_renders = FILLER(encoded_renders)
        decoded_fake_one_hot = decoder(filled_renders)
        decoded_real_one_hot = decoder(real_images)
        enc_loss = LOSS_ENCODER(real_labels, decoded_fake_one_hot, encoded_renders)
        dec_loss = LOSS_DECODER(real_labels, decoded_real_one_hot, decoded_fake_one_hot)

    tf.print(enc_loss, output_stream=LOSS_ENCODER_STREAM)
    tf.print(dec_loss, output_stream=LOSS_DECODER_STREAM)

    gradients_of_encoder = enc_tape.gradient(enc_loss, encoder.trainable_variables)
    gradients_of_decoder = dec_tape.gradient(dec_loss, decoder.trainable_variables)

    ENCODER_OPTIMISER.apply_gradients(zip(gradients_of_encoder, encoder.trainable_variables))
    DECODER_OPTIMISER.apply_gradients(zip(gradients_of_decoder, decoder.trainable_variables))


CHECKPOINT = tf.train.Checkpoint(
    encoder=encoder,
    decoder=decoder,
    encoder_optimiser=ENCODER_OPTIMISER,
    decoder_optimiser=DECODER_OPTIMISER,
)

DISPLAY_LABELS = tf.convert_to_tensor(range(10))


def save_checkpoint(epoch: int) -> None:
    """ Saves the encoder model and generates digit images for a single epoch.
    """
    CHECKPOINT.save(file_prefix=(Path.CHECKPOINT.value + 'checkpoint_'))

    encodings = encoder(DISPLAY_LABELS, training=False)

    fig, axes = plt.subplots(1, 10, figsize=(10, 1))
    for i in range(10):
        axes[i].imshow(GRID.render_values_tensor(encodings[i].numpy()), cmap='gray', vmin=-1, vmax=1)
        axes[i].set_title(i)
        axes[i].axis('off')
    plt.suptitle(f"Epoch {epoch}", y=1.3)

    plt.savefig(Path.ENCODER_IMAGE.value + f'epoch_{epoch:02d}.png')
    plt.tight_layout()
    plt.close()


def train() -> None:
    for epoch in range(NUM_EPOCHS):
        start = time.time()
        print(f'Starting epoch {epoch} at {time.asctime()}')

        for batch in range(NUM_BATCHES):
            i_min = batch * BATCH_SIZE
            i_max = (batch + 1) * BATCH_SIZE

            real_images_slice = tf.cast(MNIST.images[i_min:i_max], tf.float32)
            real_labels_slice = tf.cast(MNIST.labels[i_min:i_max], tf.int32)

            train_step(real_images_slice, real_labels_slice)

        save_checkpoint(epoch)
        print(f'Time for epoch {epoch} is {time.time() - start} seconds.')


# Run the training loop
train()


# Save the final trained encoder
encoder.save(Path.ENCODER_TRAINED.value)
