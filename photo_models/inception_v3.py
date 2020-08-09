# This basis of this model is taken from the Tensor Image classification tutorial
# https://keras.io/api/applications/  "Fine-tune InceptionV3 on a new set of classes"
# and adapted for use here.
#
import re

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import tensorflow as tf

from misc import get_optimiser, get_loss, check_model_misc_args
from photo_models.model_args import ModelArgs
from photo_models.model_misc import model_fit


def inception_v3_eg_v2(model_args: ModelArgs, verbose: bool = False):

    misc_args = check_model_misc_args(model_args.misc_args)

    # create the base pre-trained model
    # https://keras.io/api/applications/inceptionv3/
    base_model = InceptionV3(weights='imagenet', include_top=False,
                             classes=model_args.class_count,
                             input_shape=model_args.input_shape)

    # freeze the base model
    base_model.trainable = False

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # add dropout to reduce overfitting
    x = Dropout(misc_args['gsap_dropout'])(x)
    # add a fully-connected layer
    x = Dense(misc_args['gsap_units'], activation=misc_args['gsap_activation'])(x)
    x = Dropout(misc_args['gsap2_dropout'])(x)
    # add a fully-connected layer
    x = Dense(misc_args['gsap2_units'], activation=misc_args['gsap_activation'])(x)
    # and a logistic layer
    predictions = Dense(model_args.class_count, activation=misc_args['log_activation'])(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions, name=f"{base_model.name}_tl")

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    with tf.device(model_args.device_name):

        # training run 1
        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer=get_optimiser(misc_args['run1_optimizer']),
                      loss=get_loss(misc_args['run1_loss']),
                      metrics=['accuracy'])

        # train the model on the new data for a few epochs
        model_fit(model, model_args, verbose=verbose)

        # at this point, the top layers are well trained and we can start fine-tuning
        # convolutional layers from inception V3. We will freeze the bottom N layers
        # and train the remaining top layers.

        blocks = []
        # keras uses the names 'mixed<num>' for the inception blocks
        for i, layer in enumerate(base_model.layers):
            match = re.match(r"^mixed(\d+)$", layer.name)
            if match:
                blocks.append((i, int(match.group(1))))
        # we chose to train the top inception blocks, i.e. we will freeze
        # the layers associated with the first blocks and unfreeze the rest:
        num_to_train = misc_args['run2_inceptions_to_train']
        if num_to_train + 1 > len(blocks):
            raise ValueError(f"Inception blocks to train ({num_to_train}) exceeds number of blocks ({len(blocks)})")
        freeze_below = blocks[-num_to_train - 1][0] + 1
        if verbose:
            print(f"Training top {num_to_train} inception blocks, freeze below layer {freeze_below}")

        for layer in model.layers[:freeze_below]:
            layer.trainable = False
        for layer in model.layers[freeze_below:]:
            layer.trainable = True

        # training run 2
        # we need to recompile the model for these modifications to take effect
        model.compile(optimizer=get_optimiser(misc_args['run2_optimizer']),
                      loss=get_loss(misc_args['run2_loss']),
                      metrics=['accuracy'])

        # we train our model again (this time fine-tuning the top inception blocks
        # alongside the top Dense layers

    return model_fit(model, model_args, verbose=verbose, callbacks=model_args.callbacks)


def inception_v3_eg(model_args: ModelArgs, verbose: bool = False):

    # create the base pre-trained model
    # https://keras.io/api/applications/inceptionv3/
    base_model = InceptionV3(weights='imagenet', include_top=False,
                             classes=model_args.class_count,
                             input_shape=model_args.input_shape)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer
    predictions = Dense(model_args.class_count, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions, name=f"{base_model.name}_tl")

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    with tf.device(model_args.device_name):

        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # train the model on the new data for a few epochs
        model_fit(model, model_args, verbose=verbose)

        # at this point, the top layers are well trained and we can start fine-tuning
        # convolutional layers from inception V3. We will freeze the bottom N layers
        # and train the remaining top layers.

        # let's visualize layer names and layer indices to see how many layers
        # we should freeze:
        # for i, layer in enumerate(base_model.layers):
        #     print(i, layer.name)

        blocks = []
        # keras uses the names 'mixed<num>' for the inception blocks
        for i, layer in enumerate(base_model.layers):
            match = re.match(r"^mixed(\d+)$", layer.name)
            if match:
                blocks.append((i, int(match.group(1))))
        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 249 layers and unfreeze the rest:
        num_to_train = 2
        freeze_below = blocks[-num_to_train - 1][0] + 1

        for layer in model.layers[:freeze_below]:
            layer.trainable = False
        for layer in model.layers[freeze_below:]:
            layer.trainable = True

        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate
        from tensorflow.keras.optimizers import SGD
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',
                      metrics=['accuracy'])

        # we train our model again (this time fine-tuning the top 2 inception blocks
        # alongside the top Dense layers

        # history = model.fit(
        #     model_args.train_data,
        #     steps_per_epoch=step_size_train,  # total_train // batch_size,
        #     epochs=model_args.epochs,
        #     validation_data=model_args.val_data,
        #     validation_steps=step_size_valid  # total_val // batch_size
        # )

    return model_fit(model, model_args, verbose=verbose)
