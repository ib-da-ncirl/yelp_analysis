#!/usr/bin/env python3
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

import datetime
from math import floor
from timeit import default_timer as timer

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.python.keras import Input
from tensorflow.keras.optimizers import SGD

import os
import sys
import getopt
import re
import pathlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import namedtuple

from misc.config_reader import load_yaml
from misc.get_env import test_file_path, get_file_path
from misc.misc import less_dangerous_eval
from photo_models.model_args import ModelArgs
import photo_models

MIN_PYTHON = (3, 6)
if sys.version_info < MIN_PYTHON:
    sys.exit("Python %s.%s or later is required.\n" % MIN_PYTHON)


def get_config_options():
    ConfigOpt = namedtuple('ConfigOpt', ['short', 'long', 'desc'])
    __OPTS = {
        'h': ConfigOpt('h', 'help', 'Display usage'),
        'c': ConfigOpt('c:', 'cfg_path=', 'Specify path to configuration script'),
        'd': ConfigOpt('d:', 'dataset_path=', 'Specify path to the photo dataset csv file'),
        'p': ConfigOpt('p:', 'photo_path=', 'Specify path to the photos folder'),
        'm': ConfigOpt('m:', 'modelling_device=', 'TensorFlow preferred modelling device; e.g. /cpu:0'),
        'r': ConfigOpt('r:', 'run_model=', 'Model to run'),
        's': ConfigOpt('s:', 'source=', "Model source; 'img' = ImageDataGenerator or 'ds' = Dataset"),
    }
    return __OPTS


def get_short_opts() -> str:
    opts_lst = ''
    options = get_config_options()
    for o_key in options.keys():
        opts_lst += options[o_key].short
    return opts_lst


def get_long_opts() -> list:
    opts_lst = []
    options = get_config_options()
    for o_key in options.keys():
        if options[o_key].long is not None:
            opts_lst.append(options[o_key].long)
    return opts_lst


def get_short_opt(o_key) -> str:
    short_opt = ''
    options = get_config_options()
    if o_key in options.keys():
        short_opt = '-' + options[o_key].short
        if short_opt.endswith(':'):
            short_opt = short_opt[:-1]
    return short_opt


def get_long_opt(o_key) -> str:
    long_opt = ''
    options = get_config_options()
    if o_key in options.keys():
        long_opt = '--' + options[o_key].long
        if long_opt.endswith('='):
            long_opt = long_opt[:-1]
    return long_opt


def usage(name):
    print(f'Usage: {os.path.basename(name)}')
    options = get_config_options()
    lines = []
    short_len = 0
    long_len = 0
    for o_key in options:
        opt_info = options[o_key]
        if opt_info.short.endswith(':'):
            short_opt = opt_info.short[:-1] + ' <value>'
        else:
            short_opt = opt_info.short
        if opt_info.long.endswith('='):
            long_opt = opt_info.long[:-1] + ' <value>'
        else:
            long_opt = opt_info.long
        short_len = max(short_len, len(short_opt))
        long_len = max(long_len, len(long_opt))
        lines.append((short_opt, long_opt, opt_info.desc))
    for line in lines:
        print(f' -{line[0]:{short_len}.{short_len}s}|--{line[1]:{long_len}.{long_len}s} : {line[2]}')
    print()


def get_app_config(name, args):
    try:
        opts, args = getopt.getopt(args, get_short_opts(), get_long_opts())
    except getopt.GetoptError as err:
        print(err)
        usage(name)
        sys.exit(2)

    app_cfg_path = 'config.yaml'  # default in current folder

    #                dataset_path, photo_path, preferred_device, model, source
    cmd_line_opts = ['d', 'p', 'm', 'r', 's']
    cmd_line_args = {}
    for opt, arg in opts:
        if opt == get_short_opt('h') or opt == get_long_opt('h'):
            usage(name)
            sys.exit()
        elif opt == get_short_opt('c') or opt == get_long_opt('c'):
            app_cfg_path = arg  # use specified config file
        else:
            for key in cmd_line_opts:
                long_opt_name = get_long_opt(key)
                if opt == get_short_opt(key) or opt == long_opt_name:
                    long_opt_name = long_opt_name[2:]  # strip leading --
                    cmd_line_args[long_opt_name] = arg
                    break

    # get path to config file
    if not test_file_path(app_cfg_path):
        # no default so look for in environment or from console
        app_cfg_path = get_file_path('PHOTO_CFG', 'Photo classification configuration file')
        if app_cfg_path is None:
            exit(0)

    # load app config
    app_cfg = load_yaml(app_cfg_path)

    if app_cfg is not None:
        # override config from file with command line options
        for key, val in cmd_line_args.items():
            if key in ['dataset_path', 'photo_path']:
                app_cfg['defaults'][key] = val
            else:
                app_cfg[key] = val

        # check some basic configs exist
        for key in ['defaults', 'run_model']:  # required root level keys
            if key not in app_cfg.keys():
                raise EnvironmentError(f'Missing {key} configuration key')
    else:
        raise EnvironmentError(f'Missing configuration')

    # required default keys
    for key in ['dataset_path', 'photo_path', 'epochs', 'image_width', 'image_height', 'x_col', 'y_col',
                'color_mode', 'batch_size', 'seed']:
        if key not in app_cfg['defaults'].keys():
            raise EnvironmentError(f"Missing '{key}' configuration key")

    return app_cfg


def load_model_cfg(models, run_model, run_cfg):
    hierarchy = []
    search_model = run_model
    while search_model is not None:
        found_model = False
        for model_cfg in models:
            if model_cfg['name'] == search_model:
                hierarchy.insert(0, model_cfg['name'])  # insert at head
                found_model = True
                if 'parent' in model_cfg:
                    search_model = model_cfg['parent']  # look for parent
                else:
                    search_model = None
                break
        if not found_model:
            raise EnvironmentError(f"Unable to find configuration for '{search_model}'")

    # load model config starting at root model
    for model_name in hierarchy:
        for model_cfg in models:
            if model_cfg['name'] == model_name:
                for key, val in model_cfg.items():
                    run_cfg[key] = val
                break

    return run_cfg


def default_or_val(default, dictionary: dict, key: str):
    return default if key not in dictionary else dictionary[key]


def main():

    tf.debugging.set_log_device_placement(True)

    start = timer()

    # load app config
    app_cfg = get_app_config(sys.argv[0], sys.argv[1:])

    run_cfg = app_cfg['defaults'].copy()
    run_cfg['device_name'] = app_cfg['modelling_device']

    run_cfg = load_model_cfg(app_cfg['models'], app_cfg['run_model'], run_cfg)

    # required keys
    for key in ['function']:
        if key not in run_cfg.keys():
            raise EnvironmentError(f"Missing '{key}' configuration key")

    run_cfg['rescale'] = less_dangerous_eval(run_cfg['rescale'])

    print(f"Running {run_cfg['desc']}")

    params = {'dataset_path': run_cfg['dataset_path']}

    nrows = None
    if isinstance(run_cfg['photo_limit'], str):
        if run_cfg['photo_limit'].lower() != 'none':
            raise ValueError(f"Unrecognised value for 'photo_limit': '{run_cfg['photo_limit']}'")
    else:
        nrows = run_cfg['photo_limit']

    dataset_df = pd.read_csv(run_cfg['dataset_path'], nrows=nrows)

    image_generator_args = {}
    #                  name                  default
    for imgen_arg in [('featurewise_center', False),
                      ('samplewise_center', False),
                      ('featurewise_std_normalization', False),
                      ('samplewise_std_normalization', False),
                      ('zca_whitening', False),
                      ('zca_epsilon', 1e-06),
                      ('rotation_range', 0),
                      ('width_shift_range', 0.0),
                      ('height_shift_range', 0.0),
                      ('brightness_range', None),
                      ('shear_range', 0.0),
                      ('zoom_range', 0.0),
                      ('channel_shift_range', 0.0),
                      ('fill_mode', 'nearest'),
                      ('cval', 0.0),
                      ('horizontal_flip', False),
                      ('vertical_flip', False),
                      ('rescale', None),
                      ('data_format', None),
                      ('validation_split', 0.0)
                      # not supported as configurable arguments
                      # dtype=None
                      # preprocessing_function=None
                      ]:
        image_generator_args[imgen_arg[0]] = default_or_val(imgen_arg[1], run_cfg, imgen_arg[0])

    # Data preparation
    train_image_generator = ImageDataGenerator(**image_generator_args)
    common_arg = {
        'directory': run_cfg['photo_path'],
        'x_col': run_cfg['x_col'],
        'y_col': run_cfg['y_col'],
        'color_mode': run_cfg['color_mode'],
        'batch_size': run_cfg['batch_size'],
        'seed': run_cfg['seed'],
        'target_size': (run_cfg['image_height'], run_cfg['image_width']),
    }
    params.update(common_arg)

    if run_cfg['color_mode'] == 'grayscale':
        channels = 1
    else:  # 'rgb'/'rgba'
        channels = 3
    input_shape = (run_cfg['image_height'], run_cfg['image_width'], channels)
    input_tensor = Input(shape=input_shape)

    if app_cfg['source'] == 'img':
        train_data = train_image_generator.flow_from_dataframe(dataset_df,
                                                                   subset="training",
                                                                   shuffle=True,
                                                                   class_mode='categorical',
                                                                   **common_arg)
        params['train_images'] = len(train_data.filenames)

        val_data = train_image_generator.flow_from_dataframe(dataset_df,
                                                                 subset="validation",
                                                                 shuffle=True,
                                                                 class_mode='categorical',
                                                                 **common_arg)
        params['val_images'] = len(val_data.filenames)
    else:
        train_data = tf.data.Dataset.from_generator(
            lambda: train_image_generator.flow_from_dataframe(dataset_df,
                                                              subset="training",
                                                              shuffle=True,
                                                              class_mode='categorical',
                                                              **common_arg),
            output_types=(tf.float32, tf.float32),
            output_shapes=([run_cfg['batch_size'], run_cfg['image_height'], run_cfg['image_width'], channels],
                           [run_cfg['batch_size'], run_cfg['class_count']])
        )
        params['train_images'] = floor(len(dataset_df) * (1 - image_generator_args['validation_split']))

        val_data = tf.data.Dataset.from_generator(
            lambda: train_image_generator.flow_from_dataframe(dataset_df,
                                                              subset="validation",
                                                              shuffle=True,
                                                              class_mode='categorical',
                                                              **common_arg),
            output_types=(tf.float32, tf.float32),
            output_shapes=([run_cfg['batch_size'], run_cfg['image_height'], run_cfg['image_width'], channels],
                           [run_cfg['batch_size'], run_cfg['class_count']])
        )
        params['val_images'] = floor(len(dataset_df) * image_generator_args['validation_split'])

    model_args = ModelArgs(run_cfg['device_name'], input_shape, run_cfg['class_count'],
                           train_data, val_data, run_cfg['epochs'])
    model_args.set_split_total_batch(run_cfg['validation_split'], len(dataset_df), run_cfg['batch_size'])

    # get model function and call it
    method_to_call = getattr(photo_models, run_cfg['function'], None)
    if method_to_call is None:
        raise ValueError(f"The specified model '{run_cfg['function']} could not be found")
    history = method_to_call(model_args)

    params['duration'] = duration(start, True)

    show_results(history, run_cfg, app_cfg, params)


def show_results(history, run_cfg, app_cfg, params):
    if run_cfg['show_val_loss'] or run_cfg['save_val_loss'] or run_cfg['save_summary']:
        results_path = re.sub(r"<results_path_root>", app_cfg['results_path_root'], run_cfg['results_path'])
        results_path = re.sub(r"<model_name>", run_cfg['name'], results_path)
        timestamp = datetime.datetime.now()
        # std strftime directives
        results_path = re.sub(r"\{(.*)\}", lambda m: timestamp.strftime(m.group(1)), results_path)
        pathlib.Path(results_path).mkdir(parents=True, exist_ok=True)

        if run_cfg['show_val_loss'] or run_cfg['save_val_loss']:
            # Visualize training results
            # based on code from https://www.tensorflow.org/tutorials/images/classification
            acc = history.history['accuracy']
            val_acc = history.history['val_accuracy']

            loss = history.history['loss']
            val_loss = history.history['val_loss']

            epochs_range = range(history.params['epochs'])

            plt.figure(figsize=(8, 8))
            plt.subplot(1, 2, 1)
            plt.plot(epochs_range, acc, label='Training Accuracy')
            plt.plot(epochs_range, val_acc, label='Validation Accuracy')
            plt.legend(loc='lower right')
            plt.title('Training and Validation Accuracy')

            plt.subplot(1, 2, 2)
            plt.plot(epochs_range, loss, label='Training Loss')
            plt.plot(epochs_range, val_loss, label='Validation Loss')
            plt.legend(loc='upper right')
            plt.title('Training and Validation Loss')

            if run_cfg['show_val_loss']:
                # display val/loss graph
                plt.show()
            if run_cfg['save_val_loss']:
                # save val/loss graph image
                filepath = os.path.join(results_path, f"{run_cfg['name']}.png")
                print(f"Saving {filepath}")
                plt.savefig(filepath)

                # save val/loss data
                df = pd.DataFrame(data={
                    'accuracy': acc,
                    'val_accuracy': val_acc,
                    'loss': loss,
                    'val_loss': val_loss
                })
                filepath = os.path.join(results_path, f"{run_cfg['name']}.csv")
                print(f"Saving {filepath}")
                df.to_csv(filepath)

                # save val/loss data to overall results
                filepath = os.path.join(app_cfg['results_path_root'], f"result_log.csv")
                print(f"Updating {filepath}")
                new_file = not os.path.exists(filepath)
                with open(filepath, "w" if new_file else "a") as fh:
                    if new_file:
                        fh.write('model,datetime,accuracy,val_accuracy,loss,val_loss,'
                                 'image_height,image_width,train_images,val_images,epochs,'
                                 'duration,'
                                 'results_folder,dataset_path,photo_path\n')
                    last_result = df.iloc[[-1]]
                    fh.write(f"{run_cfg['name']},{timestamp.strftime('%Y-%m-%d %H:%M')},"
                             f"{last_result['accuracy'][0]},{last_result['val_accuracy'][0]},"
                             f"{last_result['loss'][0]},{last_result['val_loss'][0]},"
                             f"{params['target_size'][0]},{params['target_size'][1]},"
                             f"{params['train_images']},{params['val_images']},{history.params['epochs']},"
                             f"{params['duration']},"
                             f"{results_path},{params['dataset_path']},{params['directory']}\n")

        if run_cfg['save_summary']:
            filepath = os.path.join(results_path, "summary.txt")
            print(f"Saving {filepath}")
            with open(filepath, "w") as fh:
                history.model.summary(print_fn=lambda l: fh.write(l + '\n'))


def duration(start, verbose):
    seconds = timer() - start
    if verbose:
        print(f"Duration: {seconds:.1f}s")
    return seconds


if __name__ == '__main__':
    main()


# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

