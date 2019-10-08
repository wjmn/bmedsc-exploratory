import keras
import numpy as np
import cv2

def process_images(
    images, 
    x_render : int,
    y_render : int,
):
    
    # Upscale only to 80% of the render, to leave some padding around the digit
    up_x, up_y = int(y_render * (3/4)), int(x_render * (3/4))
    pad_y = (y_render - up_y) // 2
    pad_x = (x_render - up_x) // 2
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
            shift=(x_render // 5),
            axis=1,
            
        )
        for image in images
    ])
    
    # Scale the values from 0-255 to -1-1
    processed = (processed - 127.5) / 127.5
    
    return processed

def load_and_preprocess(
    x_render : int,
    y_render : int,
):
    
    (train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

    num_real, original_y, original_x = train_images.shape

    train_images = process_images(train_images, x_render, y_render)
    
    return train_images