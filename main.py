import matplotlib.pyplot as plt
import os
import tensorflow as tf
import matplotlib

IMAGE_SHAPE = (224, 224)
TRAINING_DATA_DIR = "input/training/training/"
VALID_DATRA_DIR = "input/validation/validation/"

matplotlib.style.use("ggplot")

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(TRAINING_DATA_DIR, shuffle=True, target_size=IMAGE_SHAPE)

valid_generator = datagen.flow_from_directory(VALID_DATRA_DIR, shuffle=True, target_size=IMAGE_SHAPE)

