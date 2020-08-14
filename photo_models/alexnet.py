from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Flatten, Dropout

from misc import get_optimiser
from photo_models.model_args import ModelArgs
from photo_models.model_misc import model_fit


def alexnet(model_args: ModelArgs, verbose=False):

    model = Sequential([
        # 1st Convolutional Layer
        Conv2D(filters=96, input_shape=model_args.input_shape, kernel_size=(11, 11), strides=(4, 4),
               padding='valid', activation='relu', name='conv_1'),
        # Max Pooling
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pooling_1'),
        # 2nd Convolutional Layer
        Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='valid', activation='relu', name='conv_2'),
        # Max Pooling
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pooling_2'),
        # 3rd Convolutional Layer
        Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu', name='conv_3'),
        # 4th Convolutional Layer
        Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu', name='conv_4'),
        # 5th Convolutional Layer
        Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu', name='conv_5'),
        # Max Pooling
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid', name='pooling_3'),
        # Fully Connected layer
        Flatten(name='flatten_1'),
        # 1st Fully Connected Layer
        Dense(4096, input_shape=(model_args.input_shape[0]*model_args.input_shape[1]*model_args.input_shape[2],),
              activation='relu', name='fully_connected_1'),
        # Add Dropout to prevent overfitting
        Dropout(0.4, name='dropout_1'),
        # 2nd Fully Connected Layer
        Dense(4096, activation='relu', name='fully_connected_2'),
        # Add Dropout
        Dropout(0.4, name='dropout_2'),
        # 3rd Fully Connected Layer
        Dense(1000, activation='relu', name='fully_connected_3'),
        # Add Dropout
        Dropout(0.4, name='dropout_3'),
        # Output Layer
        Dense(model_args.class_count, activation='softmax', name='output')
    ])

    with tf.device(model_args.device_name):
        print(f"Using '{model_args.device_name}'")

        # training run 1
        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer=get_optimiser(model_args.misc_args['run1_optimizer']),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

    return model_fit(model, model_args, verbose=verbose)
