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

"""
A collections of miscellaneous TensorFlow related functions
"""
import os
import re
from collections import namedtuple
from typing import Union

import tensorflow as tf
from numpy.core.multiarray import ndarray
from tensorflow.keras.optimizers import Adadelta, Adagrad, Adam, Adamax, Ftrl, Nadam, RMSprop, SGD
from tensorflow.python.client import device_lib
from tensorflow.python.keras.callbacks import ProgbarLogger
from tensorflow.python.keras.layers import Conv2D, Dense, Dropout, MaxPooling2D
from tensorflow.python.keras.losses import BinaryCrossentropy, CategoricalCrossentropy, SparseCategoricalCrossentropy, \
    MeanSquaredError
from tensorflow.python.keras.models import Model

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocess_input
from keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input

from misc import decode_int_or_tuple

DeviceDetail = namedtuple('DeviceDetail', ['name', 'type', 'num', 'mem'])

TF_DEV_REGEX = re.compile(r"^/(device:|physical_device:)?(cpu|gpu):(\d+)")


def get_devices():
    """
    Get TensorFlow devices
    :return:
    """
    devices = {'gpu': [], 'cpu': []}
    for device in device_lib.list_local_devices():
        dev_type = device.device_type.lower()
        if dev_type == 'gpu' or dev_type == 'cpu':
            # e.g.       name: "/device:GPU:0"  memory_limit: 1354235904
            req = TF_DEV_REGEX.match(device.name.lower())
            # e.g.                              "/device:gpu:0" "gpu"         0                 1354235904
            devices[dev_type].append(DeviceDetail(device.name, req.group(2), int(req.group(3)), device.memory_limit))
    return devices


def pick_device(dev):
    """
    Pick the device to use
    :param dev: Suggested device
    :return:
    """

    device_selection = None

    req = TF_DEV_REGEX.match(dev)
    if req:
        dtype = req.group(2)
        dnum = int(req.group(3))
    else:
        dtype = None
        dnum = None

    devices = get_devices()

    fallback = False
    if dtype is not None:
        for entry in devices[dtype]:
            if entry.num == dnum:
                device_selection = entry.name
                break

    if device_selection is None:
        max_mem = 0
        for processor in devices:
            for entry in devices[processor]:
                if entry.mem > max_mem:
                    max_mem = entry.mem
                    device_selection = entry.name
                    fallback = True

    return device_selection, False if dev == 'auto' else fallback


def restrict_gpu_mem(num, memory_limit):
    """
    Restrict TensorFlow to only allocate specified memory on the specified GPU
    :param num: GPU number
    :param memory_limit: Memory limit in MB
    :return:
    """

    if memory_limit < 1:
        raise NotImplemented("Setting % of memory doesn't seem to work ATM")
        # Restrict TensorFlow to only allocate percent of memory on GPU
        for entry in get_devices()['gpu']:
            if entry.num == num:
                memory_limit = (entry.mem * memory_limit) // 1000000
                break

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        if len(gpus) > num:
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[num],
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)
        else:
            print(f"Physical GPU {num} does not exist, ignoring set max memory request")


def set_memory_growth(enable):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, enable)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def get_optimiser(setting: Union[str, dict]):

    # 'adadelta', 'adagrad', 'adam', 'adamax', 'ftrl', 'nadam' , 'rmsprop' or 'sgd'
    # or a dict with 'name' and other args
    if isinstance(setting, str):
        optimiser = setting     # just return the name and let keras handle the rest
    elif isinstance(setting, dict):
        # need to instantiate
        name = setting['name'].lower()
        args = {key: val for key, val in setting.items() if key != 'name'}
        if name == 'adadelta':
            optimiser = Adadelta(**args)
        elif name == 'adagrad':
            optimiser = Adagrad(**args)
        elif name == 'adam':
            optimiser = Adam(**args)
        elif name == 'adamax':
            optimiser = Adamax(**args)
        elif name == 'ftrl':
            optimiser = Ftrl(**args)
        elif name == 'nadam':
            optimiser = Nadam(**args)
        elif name == 'rmsprop':
            optimiser = RMSprop(**args)
        elif name == 'sgd':
            optimiser = SGD(**args)
        else:
            raise ValueError(f"Unknown optimiser: {setting['name']}")
    else:
        raise ValueError(f"Unknown value for setting: {setting}")

    return optimiser


def get_loss(setting: Union[str, dict]):

    # 'binary_crossentropy', 'categorical_crossentropy', 'sparse_categorical_crossentropy' or 'mean_squared_error'
    # or a dict with 'name' and other args
    if isinstance(setting, str):
        loss = setting     # just return the name and let keras handle the rest
    elif isinstance(setting, dict):
        # need to instantiate
        name = setting['name'].lower()
        args = {key: val for key, val in setting.items() if key != 'name'}
        if name == 'binary_crossentropy':
            # only two label classes (assumed to be 0 and 1)
            loss = BinaryCrossentropy(**args)
        elif name == 'categorical_crossentropy':
            # labels to be provided in a one_hot representation
            loss = CategoricalCrossentropy(**args)
        elif name == 'sparse_categorical_crossentropy':
            # labels to be provided as integers
            loss = SparseCategoricalCrossentropy(**args)
        elif name == 'mean_squared_error':
            loss = MeanSquaredError(**args)
        else:
            raise ValueError(f"Unknown loss function: {setting['name']}")
    else:
        raise ValueError(f"Unknown value for setting: {setting}")

    return loss


def args_ex(args, ex_list=None):
    if ex_list is None:
        ex_list = []
    return {key: val for key, val in args.items() if key not in ex_list}


def get_conv2d(args, input_shape=None):
    # filters and kernel are positional args
    kwargs = args_ex(args, ex_list=['filters', 'kernel'])
    if input_shape is not None:
        kwargs['input_shape'] = input_shape
    return Conv2D(args['filters'], args['kernel'], **kwargs)


def get_dense(args):
    # units is positional arg
    kwargs = args_ex(args, ex_list=['units'])
    return Dense(args['units'], **kwargs)


def get_dropout(args):
    # rate is positional arg
    kwargs = args_ex(args, ex_list=['rate'])
    return Dropout(args['rate'], **kwargs)


def get_pooling(args):
    # pool_size is positional arg
    kwargs = args_ex(args, ex_list=['pool_size'])
    if 'pool_size' in args.keys():
        if isinstance(args['pool_size'], str):
            pool_size = decode_int_or_tuple(args['pool_size'])
        elif isinstance(args['pool_size'], int):
            pool_size = (args['pool_size'], args['pool_size'])
        else:
            pool_size = None
        if pool_size is None:
            print(f"Warning: Ignoring invalid 'pool_size' argument: {args['pool_size']}")
    else:
        pool_size = (2, 2)  # default from code

    return MaxPooling2D(pool_size, **kwargs)


def predict(photo_path: str, photo_file: str, target_size: tuple, model: Model, class_indices: dict, top=1):

    photo_path = os.path.join(photo_path, photo_file)

    # load image from file
    image = load_img(photo_path, target_size=target_size)
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the model
    if model.name.startswith('inception_v3'):
        image = inception_v3_preprocess_input(image)
    elif model.name.startswith('resnet50'):
        image = resnet50_preprocess_input(image)
    else:
        raise NotImplementedError(f"Preprocessing not implemented for {model.name}")

    # predict the probability across all output classes
    preds = model.predict(image)
    # convert the probabilities to class labels
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:]
        result = [(class_indices[i], pred[i]) for i in top_indices]
        results.append(result)

    return results


def preprocess_predict_img(image: ndarray, model: Model):

    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the model
    if model.name.startswith('inception_v3'):
        image = inception_v3_preprocess_input(image)
    elif model.name.startswith('resnet50'):
        image = resnet50_preprocess_input(image)
    elif model.name.startswith('tf_image'):
        pass   # no preprocessing
    else:
        raise NotImplementedError(f"Preprocessing not implemented for {model.name}")

    return image


def predict_img(image: ndarray, model: Model, classes: Union[dict, list] = None, top=1):

    image = preprocess_predict_img(image, model)

    # predict the probability across all output classes
    preds = model.predict(image, verbose=1, callbacks=[
        # ProgbarLogger()
    ])
    # convert the probabilities to class labels
    return probability_to_class(preds, classes=classes, top=top)


Prediction = namedtuple('Prediction', ['class_spec', 'probability'])


def probability_to_class(predictions: ndarray, classes: Union[dict, list] = None, top=1) -> list:

    # convert the probabilities to class labels
    results = []
    for pred in predictions:
        top_indices = pred.argsort()[-top:]     # sort indices taking top few
        if classes is not None:
            result = [Prediction(classes[i], pred[i]) for i in top_indices]
        else:
            result = pred
        results.append(result)

    return results
