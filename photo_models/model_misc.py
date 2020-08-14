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

from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.models import Model

from photo_models.model_args import ModelArgs, SplitTotalBatch
import tensorflow as tf
from keras_preprocessing.image import DataFrameIterator


def calc_step_size(data: Union[DataFrameIterator, tf.data.Dataset], info: SplitTotalBatch, set_name: Union[str, None]):
    if isinstance(data, DataFrameIterator):
        step_size = data.n // data.batch_size
        if (step_size * data.batch_size) < data.n:
            step_size += 1
    else:
        if set_name is not None:
            percent = info.val_split if set_name == 'validation' else 1 - info.val_split
        else:
            percent = 1
        step_size = (info.total * percent) // info.batch
        if (step_size * info.batch) < info.total:
            step_size += 1
    return step_size


def model_fit(model: Model, model_args: ModelArgs, callbacks=None, verbose: bool = False):

    # if verbose:
    #     model.summary()

    if callbacks is None:
        callbacks = []
    callbacks.append(EarlyStopping(monitor='loss', patience=3))

    stb = model_args.split_total_batch()
    step_size_train = calc_step_size(model_args.train_data, stb, 'training')
    step_size_valid = calc_step_size(model_args.val_data, stb, 'validation')

    # train the model on the new data for a few epochs
    history = model.fit(
        model_args.train_data,
        steps_per_epoch=step_size_train,  # total_train // batch_size,
        epochs=model_args.epochs,
        validation_data=model_args.val_data,
        validation_steps=step_size_valid,   # total_val // batch_size
        batch_size=model_args.batch_size,
        callbacks=callbacks
    )

    return history
