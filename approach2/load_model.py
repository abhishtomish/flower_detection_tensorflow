import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_model():

    test_datagen = ImageDataGenerator(rescale=1/255.)

    test_data_generator = test_datagen.flow_from_directory(
        '../flower_photos/test',
        target_size=(180, 180),
        batch_size=128,
        class_mode='categorical'
    )

    model = tf.keras.models.load_model('model.h5')

    print(model.evaluate(test_data_generator))

    test_img = "../images/rose1.jpg"
    # sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=rose_url)

    img = tf.keras.preprocessing.image.load_img(
        test_img, target_size=(180, 180)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(f"predictions - {predictions}")

    class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

    print(
        f"This image most likely belongs to {class_names[np.argmax(score)]} with a {100 * np.max(score)} percent confidence.")


if __name__ == '__main__':
    load_model()
