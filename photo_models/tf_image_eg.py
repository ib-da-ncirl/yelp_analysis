# This model is taken from the Tensor Image classification tutorial
# https://www.tensorflow.org/tutorials/images/classification
# and adapted for use here.
#
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential

from misc import get_loss, get_optimiser, get_conv2d, get_dense
from photo_models.model_args import ModelArgs
from photo_models.model_misc import model_fit


def tf_image_eg(model_args: ModelArgs, verbose: bool = False):

    misc_args = model_args.misc_args
    for arg in ['conv2D_1', 'conv2D_2', 'conv2D_3', 'dense_1',
                'run1_optimizer', 'run1_loss']:
        if arg not in misc_args:
            raise ValueError(f"Missing {arg} argument")

    # The model consists of three convolution blocks with a max pool layer in each of them.
    # There's a fully connected layer with 512 units on top of it that is activated by a relu activation function
    model = Sequential([
        get_conv2d(misc_args['conv2D_1'], model_args.input_shape),
        # Max pooling operation for 2D spatial data
        MaxPooling2D(),
        # 2D convolution layer
        get_conv2d(misc_args['conv2D_2'], model_args.input_shape),
        # Max pooling operation for 2D spatial data
        MaxPooling2D(),
        # 2D convolution layer
        get_conv2d(misc_args['conv2D_3'], model_args.input_shape),
        # Max pooling operation for 2D spatial data
        MaxPooling2D(),
        # Flattens the input. Does not affect the batch size.
        Flatten(),
        # regular densely-connected NN layer
        get_dense(misc_args['dense_1']),
        Dense(model_args.class_count)
    ], name='tf_image_tl')

    with tf.device(model_args.device_name):
        # Compile the model
        model.compile(optimizer=get_optimiser(misc_args['run1_optimizer']),
                      loss=get_loss(misc_args['run1_loss']),
                      metrics=['accuracy'])

        history = model_fit(model, model_args, verbose=verbose, callbacks=model_args.callbacks)

        # can also use model.evaluate, but it gives slight differences in values (2nd decimal place) to history
        # x = model.evaluate(x=model_args.val_data, steps=step_size_valid)

    return history
