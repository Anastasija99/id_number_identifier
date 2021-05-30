import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# from tensorflow.python.keras.optimizer_v2.rmsprop import RMSprop

if __name__ == "__main__":

    train = ImageDataGenerator(rescale=1 / 255)
    validation = ImageDataGenerator(rescale=1 / 255)
    train_dataset = train.flow_from_directory('NN_train/',
                                              target_size=(200, 200),
                                              batch_size=3,
                                              class_mode='sparse')
    val_dataset = train.flow_from_directory('NN_train/',
                                            target_size=(200, 200),
                                            batch_size=3,
                                            class_mode='sparse')

    print(f'{train_dataset.class_indices}')
    print(f'{train_dataset.classes}')

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 3)),
        tf.keras.layers.MaxPool2D(2, 2),
        #
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),
        #
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPool2D(2, 2),
        #
        tf.keras.layers.Flatten(),
        #
        tf.keras.layers.Dense(512, activation='relu'),
        #
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    # binary_crossentropy
    # categorical_crossentropy
    # RMSprop(lr=0.001)
    # "adam"
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam",
                  metrics=['accuracy'])

    model_fit = model.fit(train_dataset,
                          steps_per_epoch=3,
                          epochs=30,
                          validation_data=val_dataset
                          )

    # test
    dir_path = 'pattern/black_and_white'
    for index, path in enumerate(os.listdir(dir_path)):
        img = image.load_img(dir_path + '//' + path, target_size=(200, 200))
        X = image.img_to_array(img)
        X = np.expand_dims(X, axis=0)
        images = np.vstack([X])
        val = model.predict(images)
        plt.subplot(4, 3, index + 1)
        plt.title(f'{index} - {val[0][0]}')
        plt.imshow(img)
    plt.show()
