#  The MIT License (MIT)
#  Copyright (c) 2020. Ian Buttimer
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in all
#  copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#

# Basic model is taken from
# https://towardsdatascience.com/the-4-convolutional-neural-network-models-that-can-classify-your-fashion-images-9fe7f3e5399d
# and adapted for use here.
#
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, RandomFourierFeatures, Dropout
from tensorflow.keras.optimizers import Adam
import kerastuner.engine.hypermodel as hm

from misc import get_conv2d, get_dropout, get_dense, get_optimiser, get_loss, get_pooling
from photo_models.model_args import ModelArgs
from photo_models.model_misc import model_fit


def fashion_1(model_args: ModelArgs, verbose: bool = False):

    misc_args = model_args.misc_args
    for arg in ['conv2D_1', 'pooling_1', 'dropout_1', 'dense_1', 'log_activation',
                'run1_optimizer', 'run1_loss']:
        if arg not in misc_args:
            raise ValueError(f"Missing {arg} argument")

    model = Sequential([
        get_conv2d(misc_args['conv2D_1'], model_args.input_shape),
        get_pooling(misc_args['pooling_1']),
        get_dropout(misc_args['dropout_1']),
        Flatten(),
        get_dense(misc_args['dense_1']),
        Dense(model_args.class_count, activation=misc_args['log_activation'])
    ])

    with tf.device(model_args.device_name):
        print(f"Using '{model_args.device_name}'")
        # Compile the model
        model.compile(optimizer=get_optimiser(misc_args['run1_optimizer']),
                      loss=get_loss(misc_args['run1_loss']),
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

        model.add(Conv2D(filters=hp.Int('conv_filters', 16, 64, 8),
                         kernel_size=hp.Int('conv_kernel', 2, 4, 1), padding='same',
                         activation=hp.Choice('conv_activation', ['relu', 'softmax']),
                         input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=hp.Int('pool_filters', 2, 4, 1)))
        model.add(Dropout(rate=hp.Float('drop_rate', 0.1, 0.6, 0.1)))
        model.add(Flatten())
        model.add(Dense(units=hp.Int('dense_units', 64, 512, 32),
                        activation=hp.Choice('dense_activation', ['relu', 'softmax'])))
        model.add(Dense(self.class_count, activation='softmax'))

        model.compile(
            optimizer=Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        return model


def fashion1_tuning(model_args: ModelArgs, **kwargs):
    return Fashion1HyperModel(model_args, **kwargs)
