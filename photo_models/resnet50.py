# This model is based on the Tensor Image classification tutorial
# https://keras.io/api/applications/  "Fine-tune InceptionV3 on a new set of classes"
# and adapted for use with ResNet50 in this application.
#
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import tensorflow as tf
from tensorflow.python.keras.layers import BatchNormalization

from misc import get_optimiser, get_loss
from photo_models.model_args import ModelArgs
from photo_models.model_misc import model_fit


def resnet50_eg(model_args: ModelArgs, verbose=False):

    raise NotImplementedError("ResNet50 implementation is untested, allocation exceeds 10% of free system memory")

    misc_args = model_args.misc_args
    for arg in ['dense_1', 'log_activation',
                'run1_optimizer', 'run1_loss', 'run2_optimizer', 'run2_loss']:
        if arg not in misc_args:
            raise ValueError(f"Missing {arg} argument")

    # create the base pre-trained model
    # https://keras.io/api/applications/resnet/
    base_model = ResNet50(weights='imagenet', include_top=False,
                          classes=model_args.class_count,
                          input_shape=model_args.input_shape)

    # freeze the base model
    base_model.trainable = False

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(**misc_args['dense_1'])(x)
    # and a logistic layer
    predictions = Dense(model_args.class_count, activation=model_args.misc_args['log_activation'])(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions, name=f"{base_model.name}_tl")

    # first: train only the top layers (which were randomly initialized)
    for layer in base_model.layers:
        layer.trainable = False

    with tf.device(model_args.device_name):

        # training run 1
        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer=get_optimiser(model_args.misc_args['run1_optimizer']),
                      loss=get_loss(model_args.misc_args['run1_loss']),
                      metrics=['accuracy'])

        # train the model on the new data for a few epochs
        model_fit(model, model_args, verbose=verbose)

        # at this point, the top layers are well trained and we can start fine-tuning
        # using the pre-trained model causes issues with BatchNormalization layers, if the target dataset on which model
        # is being trained on is different from the originally used training dataset. This is because the BN layer would
        # be using the statistics from training data, instead of from inference.
        # Keras PR- https://github.com/keras-team/keras/pull/9965

        if 'run2_train_bn' in model_args.misc_args.keys() and model_args.misc_args['run2_train_bn']:
            # train BatchNormalization layers
            for layer in base_model.layers:
                layer.trainable = isinstance(layer, BatchNormalization)

        # we need to recompile the model for these modifications to take effect
        model.compile(optimizer=get_optimiser(model_args.misc_args['run2_optimizer']),
                      loss=get_loss(model_args.misc_args['run2_loss']),
                      metrics=['accuracy'])

        # we train our model again

    return model_fit(model, model_args, verbose=verbose)
