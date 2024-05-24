import matplotlib.pyplot as plt
import os
import tensorflow as tf
import matplotlib

# Constantes
IMAGE_SHAPE = (224, 224)
TRAINING_DATA_DIR = "input/training/training/"
VALID_DATRA_DIR = "input/validation/validation/"
EPOCHS = 32
BATCH_SIZE = 32

def build_model(num_classes : int):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), activation='relu',
                               input_shape=(224, 224, 3)),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])
    return model

def save_plots(train_acc, valid_acc, train_loss, valid_loss):
    """
    Função para salvar os plots de perda e acuracia
    """

    # plots de precisão
    plt.figure(figsize=(12, 9))
    plt.plot(
        train_acc, color="green", linestyle='-',
        label="acuracia do treino"
    )

    plt.plot(
        valid_acc, color="blue", linestyle='-', 
        label="acuracia da validação"
        )

    plt.xlabel("Epocas")
    plt.ylabel("Acuracia")
    plt.legend()
    plt.savefig("acuracia.png")

    # Plot de perda
    plt.figure(figsize=(12, 9))
    plt.plot(
        train_loss, color="orange", linestyle="-",
        label="perda no treino"
    )

    plt.plot(valid_loss, color="red", linestyle='-',
             label="perda na validação")

    plt.xlabel("Epocas")
    plt.ylabel("Perdas")
    plt.legend()
    plt.savefig("perda.png")
    plt.show()    

matplotlib.style.use("ggplot")

datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(TRAINING_DATA_DIR, shuffle=True, target_size=IMAGE_SHAPE)

valid_generator = datagen.flow_from_directory(VALID_DATRA_DIR, shuffle=True, target_size=IMAGE_SHAPE)

model = build_model(num_classes=10)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy']
              )
print(model.summary())

history = model.fit(train_generator,
                    steps_per_epoch=train_generator.samples // BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=valid_generator,
                    validation_steps=valid_generator.samples // BATCH_SIZE,
                    verbose=1
                    )

train_loss = history.history['loss']
train_acc = history.history['accuracy']
valid_loss = history.history['val_loss']
valid_acc = history.history['val_accuracy']

save_plots(train_acc, valid_acc, train_loss, valid_loss)