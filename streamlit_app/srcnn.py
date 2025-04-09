import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.optimizers import Adam

# SRCNN model architecture
def SRCNN():
    model = Sequential()
    model.add(Input(shape=(None, None, 1)))
    model.add(Conv2D(filters=128, kernel_size=(9, 9), kernel_initializer='glorot_uniform',
                     activation='relu', padding='valid', use_bias=True))

    model.add(Conv2D(filters=64, kernel_size=(3, 3), kernel_initializer='glorot_uniform',
                     activation='relu', padding='same', use_bias=True))

    model.add(Conv2D(filters=1, kernel_size=(5, 5), kernel_initializer='glorot_uniform',
                     activation='linear', padding='valid', use_bias=True))

    adam = Adam(learning_rate=0.0003)
    model.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])

    return model

#model and weights
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "SRCNN", "weights.h5")
srcnn_model = SRCNN()
srcnn_model.load_weights(model_path)

def process_srcnn(image):
    """
    Process the image using SRCNN.
    Args:
    - image: Low-resolution input image (numpy array).
    
    Returns:
    - High-resolution output image (numpy array).
    """
    if len(image.shape) == 2 or image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y_channel = ycbcr[:, :, 0].astype(np.float32) / 255.0

    input_tensor = np.expand_dims(np.expand_dims(y_channel, axis=0), axis=-1)

    predicted_y = srcnn_model.predict(input_tensor, verbose=0)[0, :, :, 0]
    predicted_y = (predicted_y * 255).clip(0, 255).astype(np.uint8)

    h, w = predicted_y.shape
    ycbcr = ycbcr[0:h, 0:w, :]
    ycbcr[:, :, 0] = predicted_y

    output_image = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2BGR)

    return output_image
