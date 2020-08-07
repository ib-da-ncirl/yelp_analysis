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
from typing import Union

from tensorflow.python.keras.models import Model

from photo_models.model_args import ModelArgs, SplitTotalBatch
import tensorflow as tf
from keras_preprocessing.image import DataFrameIterator


def run_and_history(model, model_args: ModelArgs):
    with tf.device(model_args.device_name):
        model.summary()

        step_size_train = model_args.train_data.n // model_args.train_data.batch_size
        step_size_valid = model_args.val_data.n // model_args.val_data.batch_size

        history = model.fit(
            model_args.train_data,
            steps_per_epoch=step_size_train,  # total_train // batch_size,
            epochs=model_args.epochs,
            validation_data=model_args.val_data,
            validation_steps=step_size_valid  # total_val // batch_size
        )

    return model, history


def calc_step_size(data: Union[DataFrameIterator, tf.data.Dataset], info: SplitTotalBatch, set_name: str):
    if isinstance(data, DataFrameIterator):
        step_size = data.n // data.batch_size
    else:
        percent = info.val_split if set_name == 'validation' else 1 - info.val_split
        step_size = (info.total * percent) // info.batch
    return step_size


def model_fit(model: Model, model_args: ModelArgs, verbose: bool = False):

    if verbose:
        model.summary()

    stb = model_args.split_total_batch()
    step_size_train = calc_step_size(model_args.train_data, stb, 'training')
    step_size_valid = calc_step_size(model_args.val_data, stb, 'validation')

    # train the model on the new data for a few epochs
    history = model.fit(
        model_args.train_data,
        steps_per_epoch=step_size_train,  # total_train // batch_size,
        epochs=model_args.epochs,
        validation_data=model_args.val_data,
        validation_steps=step_size_valid  # total_val // batch_size
    )

    return history
