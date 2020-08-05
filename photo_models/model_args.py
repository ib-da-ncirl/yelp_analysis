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
from collections import namedtuple

from keras_preprocessing.image import DataFrameIterator
import tensorflow as tf
from tensorflow.keras import Input

SplitTotalBatch = namedtuple('SplitTotalBatch', ['val_split', 'total', 'batch'])


class ModelArgs:

    def __init__(self, device_name: str, input_shape: tuple, input_tensor: Input, class_count: int,
                 train_data: Union[DataFrameIterator, tf.data.Dataset],
                 val_data: Union[DataFrameIterator, tf.data.Dataset],
                 epochs: int, misc_args=None):
        """
        Initialise this object
        :param device_name: TF device name to use
        :param input_shape: Shape of input images
        :param input_tensor: Input tensor
        :param class_count: Number of classification classes
        :param train_data: Training data
        :param val_data: Validation data
        :param epochs: Number of epochs
        """
        if misc_args is None:
            misc_args = {}
        self._device_name = device_name
        self._input_shape = input_shape
        self._input_tensor = input_tensor
        self._class_count = class_count
        self._train_data = train_data
        self._val_data = val_data
        self._epochs = epochs
        self._misc_args = misc_args
        self._validation_split = 0.0
        self._total = 0
        self._batch_size = 0

    @property
    def device_name(self) -> str:
        return self._device_name

    @device_name.setter
    def device_name(self, device_name: str):
        self._device_name = device_name

    @property
    def input_shape(self) -> tuple:
        return self._input_shape

    @input_shape.setter
    def input_shape(self, input_shape: tuple):
        self._input_shape = input_shape

    @property
    def input_tensor(self) -> Input:
        return self._input_tensor

    @input_tensor.setter
    def input_tensor(self, input_tensor: Input):
        self._input_tensor = input_tensor

    @property
    def class_count(self) -> int:
        return self._class_count

    @class_count.setter
    def class_count(self, class_count: int):
        self._class_count = class_count

    @property
    def train_data(self) -> Union[DataFrameIterator, tf.data.Dataset]:
        return self._train_data

    @train_data.setter
    def train_data(self, train_data: Union[DataFrameIterator, tf.data.Dataset]):
        self._train_data = train_data

    @property
    def val_data(self) -> Union[DataFrameIterator, tf.data.Dataset]:
        return self._val_data

    @val_data.setter
    def val_data(self, val_data: Union[DataFrameIterator, tf.data.Dataset]):
        self._val_data = val_data

    @property
    def epochs(self) -> int:
        return self._epochs

    @epochs.setter
    def epochs(self, epochs: int):
        self._epochs = epochs

    @property
    def misc_args(self) -> dict:
        return self._misc_args

    @misc_args.setter
    def misc_args(self, misc_args: dict):
        self._misc_args = misc_args

    @property
    def validation_split(self) -> float:
        return self._validation_split

    @validation_split.setter
    def validation_split(self, validation_split: float):
        self._validation_split = validation_split

    @property
    def total(self) -> int:
        return self._total

    @total.setter
    def total(self, total: int):
        self._total = total

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        self._batch_size = batch_size

    def set_split_total_batch(self, validation_split: float, total: int, batch_size: int):
        self._validation_split = validation_split
        self._total = total
        self._batch_size = batch_size

    def split_total_batch(self) -> SplitTotalBatch:
        return SplitTotalBatch(self._validation_split, self._total, self._batch_size)
