# This model is taken from the Tensor Image classification tutorial
# https://www.tensorflow.org/tutorials/images/classification
# and adapted for use here.
#
import tensorflow as tf
from keras import regularizers
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Activation, RandomFourierFeatures
from tensorflow.keras.optimizers import Adam
import kerastuner.engine.hypermodel as hm

from misc import get_loss, get_optimiser, get_conv2d, get_dense
from photo_models.model_args import ModelArgs
from photo_models.model_misc import model_fit


def tf_image_eg(model_args: ModelArgs, verbose: bool = False):

    misc_args = model_args.misc_args

    # The model consists of three convolution blocks with a max pool layer in each of them.
    # There's a fully connected layer with 512 units on top of it that is activated by a relu activation function
    model = Sequential(
        _get_layers(model_args, include_top=True, verbose=verbose), name='tf_image_tl')

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


def _get_layers(model_args: ModelArgs, include_top=True, verbose: bool = False):

    misc_args = model_args.misc_args
    for arg in ['conv2D_1', 'conv2D_2', 'conv2D_3', 'dense_1',
                'run1_optimizer', 'run1_loss']:
        if arg not in misc_args:
            raise ValueError(f"Missing {arg} argument")
    # The model consists of three convolution blocks with a max pool layer in each of them.

    model = [
        get_conv2d(misc_args['conv2D_1'], model_args.input_shape),
        # Max pooling operation for 2D spatial data
        MaxPooling2D(),
        # 2D convolution layer
        get_conv2d(misc_args['conv2D_2']),
        # Max pooling operation for 2D spatial data
        MaxPooling2D(),
        # 2D convolution layer
        get_conv2d(misc_args['conv2D_3']),
        # Max pooling operation for 2D spatial data
        MaxPooling2D(),
        # Flattens the input. Does not affect the batch size.
        Flatten(),
        # regular densely-connected NN layer
        get_dense(misc_args['dense_1'])
    ]
    if include_top:
        model.append(Dense(model_args.class_count))

    return model


class TfImageHyperModel(hm.HyperModel):
    """
    Keras Tuner HyperModel
    """
    def __init__(self, model_args: ModelArgs, **kwargs):
        super(TfImageHyperModel, self).__init__(**kwargs)
        self.input_shape = model_args.input_shape
        self.class_count = model_args.class_count

    def build(self, hp):
        model = Sequential()

        model.add(Conv2D(filters=hp.Int('filters', 16, 64, 8), kernel_size=3, padding='same',
                         activation=hp.Choice('activation', ['relu', 'softmax']),
                         input_shape=self.input_shape))
        model.add(MaxPooling2D())
        for i in range(hp.Int('num_layers', 1, 10)):
            model.add(Conv2D(filters=hp.Int('filters', 32, 64, 8), kernel_size=3, padding='same',
                             activation=hp.Choice('activation', ['relu', 'softmax'])))
            model.add(MaxPooling2D())
        model.add(Flatten())
        model.add(Dense(units=hp.Int('units', 64, 512, 32),
                        activation=hp.Choice('activation', ['relu', 'softmax'])))
        model.add(Dense(self.class_count, activation='softmax'))

        model.compile(
            optimizer=Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
            loss='categorical_crossentropy',
            metrics=['accuracy'])
        return model


def tf_image_tuning(model_args: ModelArgs, **kwargs):
    return TfImageHyperModel(model_args, **kwargs)


def tf_image_eg_qsvm(model_args: ModelArgs, verbose: bool = False):

    misc_args = model_args.misc_args

    # The model consists of three convolution blocks with a max pool layer in each of them.
    # There's a fully connected layer with 512 units on top of it that is activated by a relu activation function
    model = Sequential(
        _get_layers(model_args, include_top=False, verbose=verbose), name='tf_image_tl_qsvm')

    model.add(RandomFourierFeatures(
        output_dim=4096, scale=10.0, kernel_initializer="gaussian"
    ))
    model.add(Dense(model_args.class_count, kernel_regularizer=regularizers.l2(0.0001)))
    model.add(Activation('linear'))

    with tf.device(model_args.device_name):
        print(f"Using '{model_args.device_name}'")

        # Compile the model
        # model.compile(optimizer=get_optimiser(misc_args['run1_optimizer']),
        #               loss=get_loss(misc_args['run1_loss']),
        #               metrics=['accuracy'])
        model.compile(loss='squared_hinge',
                      optimizer='adadelta', metrics=['accuracy'])

        history = model_fit(model, model_args, verbose=verbose, callbacks=model_args.callbacks)

        # can also use model.evaluate, but it gives slight differences in values (2nd decimal place) to history
        # x = model.evaluate(x=model_args.val_data, steps=step_size_valid)



    return history
