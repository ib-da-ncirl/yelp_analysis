# This model is taken from the Tensor Image classification tutorial
# https://www.tensorflow.org/tutorials/images/classification
# and adapted for use here.
#
from photo_models.model_args import ModelArgs
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D

from photo_models.model_misc import calc_step_size


def tf_image_eg(model_args: ModelArgs):
    # The model consists of three convolution blocks with a max pool layer in each of them.
    # There's a fully connected layer with 512 units on top of it that is activated by a relu activation function
    model = Sequential([
        # filters, kernel_size
        Conv2D(16, 3, padding='same', activation='relu', input_shape=model_args.input_shape),
        MaxPooling2D(),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(model_args.class_count)
    ])

    with tf.device(model_args.device_name):
        # Compile the model
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        model.summary()

        stb = model_args.split_total_batch()
        step_size_train = calc_step_size(model_args.train_data, stb, 'training')
        step_size_valid = calc_step_size(model_args.val_data, stb, 'validation')

        history = model.fit(
            model_args.train_data,
            steps_per_epoch=step_size_train,  # total_train // batch_size,
            epochs=model_args.epochs,
            validation_data=model_args.val_data,
            validation_steps=step_size_valid  # total_val // batch_size
        )

        # can also use model.evaluate, but it gives slight differences in values (2nd decimal place) to history
        # x = model.evaluate(x=model_args.val_data, steps=step_size_valid)

    return history
