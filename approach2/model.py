import tensorflow as tf

from dataset import image_generator


def create_model():

    training_data, validation_data = image_generator()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, (3, 3), input_shape=(180, 180, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ])

    print(model.summary())

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    history = model.fit(
        training_data,
        validation_data=validation_data,
        epochs=20
    )

    model.save('model.h5')

    return history


if __name__ == '__main__':
    create_model()
