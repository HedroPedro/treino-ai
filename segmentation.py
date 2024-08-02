import os
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib
import tensorflow_datasets as tfds

AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 20

class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        show_predictions()
        print(f"Prediçao de amostra apos época {epoch+1}")

def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    plt.figure(figsize=(15, 15))
    plt.tight_layout()
    plt.plot(train_acc, color="blue", linestyle="-", label="Acuracia de treino")
    plt.plot(valid_acc, color="orange", linestyle="-", label="Acuracia de validação")

    plt.xlabel("Epocas")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.savefig("acuracia.png")
    plt.show()

    plt.figure(figsize=(15, 15))
    plt.tight_layout()
    plt.plot(train_loss, color="blue", linestyle="-", label="Acuracia de treino")
    plt.plot(valid_loss, color="orange", linestyle="-", label="Acuracia de validação")

    plt.xlabel("Epocas")
    plt.ylabel("Perda")
    plt.legend()
    plt.grid(True)
    plt.savefig("perda.png")
    plt.show()

    plt.plot()

def show_predictions(dataset=None, num=1):
  if dataset:
    for image, mask in dataset.take(num):
      pred_mask = model.predict(image)
      display([image[0], mask[0], create_mask(pred_mask)])
  else:
    display([sample_image, sample_mask,
             create_mask(model.predict(sample_image[tf.newaxis, ...]))])

def normalize(img, mask):
    img = tf.cast(img, tf.float32) / 255.0
    mask -= 1
    return img, mask

def display(display_list):
  plt.figure(figsize=(15, 15))

  title = ['Image', 'Mascara', 'Mascara prevista']

  for i in range(len(display_list)):
    plt.subplot(1, len(display_list), i+1)
    plt.title(title[i])
    plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
    plt.axis('off')
  plt.show()

@tf.function
def load_image_train(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    input_image, input_mask = normalize(input_image, input_mask)

    if tf.random.normal(()) < 0.5:
      input_image = tf.image.flip_left_right(input_image)
      input_mask = tf.image.flip_left_right(input_mask)

    return input_image, input_mask

def load_image_test(datapoint):
  input_image = tf.image.resize(datapoint['image'], (128, 128))
  input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

  input_image, input_mask = normalize(input_image, input_mask)

  return input_image, input_mask

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def double_conv_block(x, n_filters):
  x = tf.keras.layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
  x = tf.keras.layers.Conv2D(n_filters, 3, padding = "same", activation = "relu", kernel_initializer = "he_normal")(x)
  return x

def upsample_block(x, conv_features, n_filters):
  x = tf.keras.layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
  x = tf.keras.layers.concatenate([x, conv_features])
  x = tf.keras.layers.Dropout(0.3)(x)
  x = double_conv_block(x, n_filters)

  return x

def build_model_seg():
  input = tf.keras.Input(shape=(128, 128, 3))
  base_model = tf.keras.applications.MobileNetV2(input_shape=(128, 128, 3), include_top=False,
                                                weights="imagenet")

  layer_names = [
    "block_1_expand_relu",
    "block_3_expand_relu",
    "block_6_expand_relu",
    "block_13_expand_relu",
    "block_16_project"
  ]

  layers = [base_model.get_layer(name).output for name in layer_names]

  down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers, name="V2")
  down_stack.trainable = False

  x = input
  skips = down_stack(x)
  x = skips[-1]
  skips = reversed(skips[:-1])

  #Bottleneck
  x = double_conv_block(x, 1024)

  i = 0
  for layer in skips:
    x = upsample_block(x, layer, int(512/(2**i)))
    i+=1

  output = tf.keras.layers.Conv2DTranspose(3, 3, 2, padding="same", activation="softmax", name="output")(x)

  return tf.keras.Model(input, output, name="U-Net")

def segmentation_ai():
    global sample_image, sample_mask, model
    dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True, data_dir="tensorflow-datasets")
    TRAIN_LENGTH = info.splits['train'].num_examples
    BATCH_SIZE = 64
    BUFFER_SIZE = 1000
    STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE
    train = dataset['train'].map(load_image_train, num_parallel_calls=AUTOTUNE)
    test = dataset['test'].map(load_image_test)
    train_dataset = train.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test.take(3000).batch(BATCH_SIZE)
    VAL_SUBSPLITS = 5
    VALIDATION_STEPS = info.splits['test'].num_examples//BATCH_SIZE//VAL_SUBSPLITS

    for image, mask in train.take(1):
      sample_image, sample_mask = image, mask

    #model = tf.keras.models.load_model("ais/segmentation.h5")
    model = build_model_seg()
    model.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    model.summary()
    history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_dataset,
                          use_multiprocessing=True)

    model.save("ais/segmentation.h5")
    
    for image, mask in train.take(1):
        final_pred_mask = create_mask(model.predict(image[tf.newaxis, ...]))
    display([sample_image, sample_mask, final_pred_mask])

    return history

history = segmentation_ai()
train_loss = history.history['loss']
train_acc = history.history['accuracy']
valid_loss = history.history['val_loss']
valid_acc = history.history['val_accuracy']
save_plots(train_acc, valid_acc, train_loss, valid_loss)