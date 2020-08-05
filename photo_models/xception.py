# This model is taken from the Transfer learning & fine-tuning Developer guide
# https://keras.io/guides/transfer_learning/
# and adapted for use here.
#
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import tensorflow as tf
from tensorflow.keras.applications.xception import Xception
from tensorflow.python.keras import Input

from misc import get_optimiser
from photo_models.model_args import ModelArgs
from photo_models.model_misc import model_fit


def xception_eg(model_args: ModelArgs, verbose: bool = False):

    raise NotImplementedError("Xception implementation is untested, memory requirements exceed availability")

    # create the base pre-trained model
    # https://keras.io/api/applications/xception/
    base_model = Xception(weights='imagenet', include_top=False,
                          classes=model_args.class_count,
                          input_shape=model_args.input_shape)

    # freeze the base model
    base_model.trainable = False

    inputs = Input(shape=model_args.input_shape)
    # We make sure that the base_model is running in inference mode here,
    # by passing `training=False`. This is important for fine-tuning, as you will
    # learn in a few paragraphs.
    x = base_model(inputs, training=False)
    # Convert features of shape `base_model.output_shape[1:]` to vectors
    x = GlobalAveragePooling2D()(x)
    # A Dense classifier with number of classes
    outputs = Dense(model_args.class_count)(x)
    model = Model(inputs, outputs)

    with tf.device(model_args.device_name):

        # training run 1
        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer=get_optimiser(model_args.misc_args['run1_optimizer']),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # train the model on the new data for a few epochs
        model_fit(model, model_args, verbose=verbose)

        # Unfreeze the base model
        base_model.trainable = True

        # training run 2
        # we need to recompile the model for these modifications to take effect
        model.compile(optimizer=get_optimiser(model_args.misc_args['run2_optimizer']),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # we train our model again (this time fine-tuning the top inception blocks
        # alongside the top Dense layers

    return model_fit(model, model_args, verbose=verbose)
