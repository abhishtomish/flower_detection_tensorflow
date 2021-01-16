import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def image_generator():
    train_datagen = ImageDataGenerator(
        rescale=1/255.,
        rotation_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    validation_datagen = ImageDataGenerator(rescale=1/255.)

    train_data_generator = train_datagen.flow_from_directory(
        '../flower_photos/train',
        target_size=(180, 180),
        batch_size=128,
        class_mode='categorical'
    )

    validation_data_generator = validation_datagen.flow_from_directory(
        '../flower_photos/validation',
        target_size=(180, 180),
        batch_size=128,
        class_mode='categorical'
    )

    return train_data_generator, validation_data_generator


if __name__ == '__main__':
    image_generator()
