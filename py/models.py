from keras.applications import *
from keras.layers import *
from keras.models import Model

def get_regression_model():
    base_model = ResNet50(include_top=False, weights="imagenet",
                          input_shape=(None, None, 3))
    final_layer = "activation_49"
    y = GlobalAveragePooling2D()(base_model.get_layer(final_layer).output)
    y = Dense(2048, activation="relu")(y)
    y = Dense(5)(y)
    model = Model(base_model.input, y)
    for layer in model.layers:
        if "dense" not in layer.name:
            layer.trainable = False
    model.compile("adam", "mse", metrics=["mae"])
    model.summary()
    return model


def cbr(N, k, s):
    def f(x):
        y = Conv2D(N, k, strides=s, padding="same")(x)
        y = BatchNormalization(axis=-1)(y)
        y = Activation("relu")(y)
        return y

    return f


def get_model(inshape=(256, 256, 1), outputs=(8, 9)):
    x = Input(inshape)

    y = cbr(32, 7, 2)(x)

    y = cbr(32, 3, 1)(y)
    y = cbr(32, 3, 1)(y)
    y = cbr(32, 3, 2)(y)

    y = cbr(64, 3, 1)(y)
    y = cbr(64, 3, 1)(y)
    y = cbr(64, 3, 2)(y)

    y = cbr(128, 3, 1)(y)
    y = cbr(128, 3, 1)(y)
    y = cbr(128, 3, 2)(y)

    y = cbr(256, 3, 1)(y)
    y = cbr(256, 3, 1)(y)
    y = cbr(256, 3, 2)(y)

    y = Flatten()(y)

    # For iso models
    y = Dense(outputs, name="block_params")(y)
    model = Model(x, y)

    # For random rotations model
    # y1 = Dense(outputs[0], name="block_params")(y)
    # y2 = Dense(outputs[1], name="rot_params")(y)
    # model = Model(x, [y1, y2])

    metrics_dict = {"block_params": "mae", "rot_params": "mae"}
    model.compile("adam", loss="mse", metrics=metrics_dict)
    model.summary()
    return model