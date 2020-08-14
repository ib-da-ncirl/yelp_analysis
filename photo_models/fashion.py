# This model is taken from
# https://towardsdatascience.com/the-4-convolutional-neural-network-models-that-can-classify-your-fashion-images-9fe7f3e5399d
# and adapted for use here.
#
import tensorflow as tf
from keras import regularizers
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, RandomFourierFeatures, Dropout
from tensorflow.keras.optimizers import Adam
import kerastuner.engine.hypermodel as hm

from photo_models.model_args import ModelArgs
from photo_models.model_misc import model_fit


def fashion_1(model_args: ModelArgs, verbose: bool = False):

    misc_args = model_args.misc_args

    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=model_args.input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(model_args.class_count, activation='softmax')
    ])

    with tf.device(model_args.device_name):
        print(f"Using '{model_args.device_name}'")
        # Compile the model
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        history = model_fit(model, model_args, verbose=verbose, callbacks=model_args.callbacks)

        # can also use model.evaluate, but it gives slight differences in values (2nd decimal place) to history
        # x = model.evaluate(x=model_args.val_data, steps=step_size_valid)

    return history


class Fashion1HyperModel(hm.HyperModel):
    """
    Keras Tuner HyperModel
    """

    def __init__(self, model_args: ModelArgs, **kwargs):
        super(Fashion1HyperModel, self).__init__(**kwargs)
        self.input_shape = model_args.input_shape
        self.class_count = model_args.class_count

    def build(self, hp):
        model = Sequential()

        model.add(Conv2D(filters=hp.Int('filters', 16, 64, 8),
                         kernel_size=hp.Int('filters', 2, 4, 1), padding='same',
                         activation=hp.Choice('activation', ['relu', 'softmax']),
                         input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=hp.Int('filters', 2, 4, 1)))
        model.add(Dropout(rate=hp.Float('rate', 0.1, 0.6, 0.1)))
        model.add(Flatten())
        model.add(Dense(units=hp.Int('units', 64, 512, 32),
                        activation=hp.Choice('activation', ['relu', 'softmax'])))
        model.add(Dense(self.class_count, activation='softmax'))

        model.compile(
            optimizer=Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        return model


def fashion1_tuning(model_args: ModelArgs, **kwargs):
    return Fashion1HyperModel(model_args, **kwargs)
