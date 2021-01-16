import tensorflow as tf

from dataset import *

num_classes = 5


def load_model():
    data_augmentation = tf.keras.models.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal",
                                                                  input_shape=(img_height,
                                                                               img_width,
                                                                               3)),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
            tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
        ]
    )

    model = tf.keras.models.Sequential([
        data_augmentation,
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(img_height, img_width, 3)),
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    model.load_weights("model.h5")

    return model
