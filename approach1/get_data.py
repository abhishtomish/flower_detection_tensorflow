import os
import PIL
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pathlib
import webbrowser

dataset_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

roses = list(data_dir.glob('roses/*'))

# webbrowser.open(roses[0])
