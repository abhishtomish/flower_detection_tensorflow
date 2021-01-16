from approach1.load_model import *
from approach1.dataset import *
from approach1.get_data import *


if __name__ == '__main__':
    # history, epochs = create_model()
    # measure_performance(history, epochs)
    model = load_model()

    test_img = "images/tulip1.jpg"
    # sunflower_path = tf.keras.utils.get_file('Red_sunflower', origin=rose_url)

    img = tf.keras.preprocessing.image.load_img(
        test_img, target_size=(img_height, img_width)
    )
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(f"This image most likely belongs to {class_names[np.argmax(score)]} with a {100 * np.max(score)} percent confidence.")
