# coding:utf8
import os
import string

import numpy as np
import tensorflow as tf

from captcha_tensorflow_apiv2 import gen_captcha


def crack_captcha_cnn(MAX_CAPTCHA, CHAR_SET_LEN):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(32, (3, 3)))
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2))

    model.add(tf.keras.layers.Conv2D(64, (5, 5)))
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2))

    model.add(tf.keras.layers.Conv2D(128, (5, 5)))
    model.add(tf.keras.layers.PReLU())
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(MAX_CAPTCHA * CHAR_SET_LEN))
    model.add(tf.keras.layers.Reshape([MAX_CAPTCHA, CHAR_SET_LEN]))

    model.add(tf.keras.layers.Softmax())

    return model


if __name__ == '__main__':
    captcha = gen_captcha.generateCaptcha()
    characters = string.digits + string.ascii_uppercase
    width, height, char_num, characters, classes = captcha.get_parameter()
    SAVE_PATH = os.path.abspath("./")
    try:
        model = tf.keras.models.load_model(SAVE_PATH + 'model')
    except Exception as e:
        print('#######Exception', e)
        model = crack_captcha_cnn(4,62)

    model.compile(optimizer='Adam',
                  metrics=['accuracy'],
                  loss='categorical_crossentropy')

    for times in range(500000):
        batch_x, batch_y = next(captcha.gen_captcha(batch_size=512))
        print('times=', times, ' batch_x.shape=', batch_x.shape, ' batch_y.shape=', batch_y.shape)
        import os

        print(os.path.abspath('./'))
        model.fit(batch_x, batch_y, epochs=4)
        print("y预测=\n", np.argmax(model.predict(batch_x), axis=2))
        print("y实际=\n", np.argmax(batch_y, axis=2))

        if 0 == times % 10:
            print("save model at times=", times)
            model.save(SAVE_PATH + 'model')
