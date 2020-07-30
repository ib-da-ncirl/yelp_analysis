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

import re
from collections import namedtuple
import tensorflow as tf
from tensorflow.python.client import device_lib

DeviceDetail = namedtuple('DeviceDetail', ['name', 'type', 'num', 'mem'])

TF_DEV_REGEX = re.compile(r"^/(device:|physical_device:){,1}(cpu|gpu):(\d+)")


def get_devices():
    """
    Get TensotFlow devices
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

