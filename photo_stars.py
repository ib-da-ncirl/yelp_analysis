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
import os
import pathlib
import re
import sys
from math import floor
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input

import photo_models
from misc import less_dangerous_eval, ArgOptParam, default_or_val, pick_device, restrict_gpu_mem, ArgCtrl
from photo_models import ModelArgs

MIN_PYTHON = (3, 6)
if sys.version_info < MIN_PYTHON:
    sys.exit("Python %s.%s or later is required.\n" % MIN_PYTHON)


def get_app_config(name: str, args: list):
    """
    Get the application config
    Note: command line options override config file options
    :param name: Name of script
    :param args: command line args
    :return:
    """
    arg_ctrl = ArgCtrl(os.path.basename(name), dflt_config='config.yaml')
    arg_ctrl.add_option('d', 'dataset_path', 'Specify path to the photo dataset csv file', has_value=True)
    arg_ctrl.add_option('p', 'photo_path', 'Specify path to the photos folder', has_value=True)
    arg_ctrl.add_option('m', 'modelling_device', 'TensorFlow preferred modelling device; e.g. /cpu:0', has_value=True)
    arg_ctrl.add_option('r', 'run_model', 'Model to run', has_value=True)
    arg_ctrl.add_option('s', 'source', "Model source; 'img' = ImageDataGenerator or 'ds' = Dataset", has_value=True)
    arg_ctrl.add_option('b', 'random_batch', "If < 1, percent of available photo to randomly sample, "
                                             "else number to randomly sample",
                        has_value=True, type=float)
    arg_ctrl.add_option('l', 'photo_limit', "Max number of photos to use; 'none' to use all available, or a number",
                        has_value=True, type=int)
    arg_ctrl.add_option('v', 'verbose', 'Verbose mode')

    app_cfg = arg_ctrl.get_app_config(args)

    # check some basic configs exist
    for key in ['defaults', 'run_model']:  # required root level keys
        if key not in app_cfg.keys():
            error(f'Missing {key} configuration key')
    # move command line args to where they should be
    for key in ['dataset_path', 'photo_path', 'photo_limit']:
        if key in app_cfg:
            app_cfg['defaults'][key] = app_cfg[key]
            app_cfg.pop(key)
    # required default keys
    for key in ['dataset_path', 'photo_path', 'epochs', 'image_width', 'image_height', 'x_col', 'y_col',
                'color_mode', 'batch_size', 'seed']:
        if key not in app_cfg['defaults'].keys():
            error(f"Missing '{key}' configuration key")

    return app_cfg


def error(msg):
    sys.exit(f"Error: {msg}")


def load_model_cfg(models: list, run_model: str, run_cfg: dict):
    """
    Load model config
    :param models: List of available models
    :param run_model: Name of model to run
    :param run_cfg: Run config
    :return:
    """
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
            error(f"Unable to find configuration for '{search_model}'")

    # load model config starting at root model
    for model_name in hierarchy:
        for model_cfg in models:
            if model_cfg['name'] == model_name:
                for key, val in model_cfg.items():
                    run_cfg[key] = val
                break

    return run_cfg


def main():
    start = timer()

    # load app config
    app_cfg = get_app_config(sys.argv[0], sys.argv[1:])

    tf.debugging.set_log_device_placement(False if 'tf_log_device_placement' not in app_cfg
                                          else app_cfg['tf_log_device_placement'])

    if 'gpu_memory_limit' in app_cfg:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        restrict_gpu_mem(0, app_cfg['gpu_memory_limit'])
        # Restrict TensorFlow to only allocate 90% of memory on the first GPU
        # restrict_gpu_mem(0, 0.75)   # doesn't seem to work

    base_cfg = app_cfg['defaults'].copy()

    base_cfg['device_name'], fallback = pick_device(app_cfg['modelling_device'])
    if fallback:
        print(f"Device '{app_cfg['modelling_device']}' not available")
    print(f"Using '{base_cfg['device_name']}'")

    if isinstance(app_cfg['run_model'], str):
        models_run_list = [app_cfg['run_model']]
    else:
        models_run_list = app_cfg['run_model']

    model_idx = 0
    for model_to_run in models_run_list:

        tf.keras.backend.clear_session()  # clear keras state

        run_cfg = base_cfg.copy()

        run_cfg = load_model_cfg(app_cfg['models'], model_to_run, run_cfg)

        # required keys
        for key in ['function']:
            if key not in run_cfg.keys():
                error(f"Missing '{key}' configuration key")

        run_cfg['rescale'] = less_dangerous_eval(run_cfg['rescale'])

        model_idx += 1
        print(f"\nRunning {run_cfg['desc']} ({model_idx/len(models_run_list)})")
        if 'verbose' in app_cfg:
            print(f"Run config: {run_cfg}")

        params = {'dataset_path': run_cfg['dataset_path']}

        nrows = None
        if isinstance(run_cfg['photo_limit'], str):
            if run_cfg['photo_limit'].lower() != 'none':
                error(f"Unrecognised value for 'photo_limit': '{run_cfg['photo_limit']}'")
        else:
            nrows = run_cfg['photo_limit']

        dataset_df = pd.read_csv(run_cfg['dataset_path'], nrows=nrows)

        if 'random_batch' in app_cfg:
            sample_ctrl = {
                'n': int(app_cfg['random_batch']) if app_cfg['random_batch'] >= 1.0 else None,
                'frac': app_cfg['random_batch'] if app_cfg['random_batch'] < 1.0 else None
            }
            dataset_df = dataset_df.sample(**sample_ctrl, random_state=1)

        image_generator_args = {}
        for imgen_arg in [ArgOptParam('featurewise_center', False),
                          ArgOptParam('samplewise_center', False),
                          ArgOptParam('featurewise_std_normalization', False),
                          ArgOptParam('samplewise_std_normalization', False),
                          ArgOptParam('zca_whitening', False),
                          ArgOptParam('zca_epsilon', 1e-06),
                          ArgOptParam('rotation_range', 0),
                          ArgOptParam('width_shift_range', 0.0),
                          ArgOptParam('height_shift_range', 0.0),
                          ArgOptParam('brightness_range', None),
                          ArgOptParam('shear_range', 0.0),
                          ArgOptParam('zoom_range', 0.0),
                          ArgOptParam('channel_shift_range', 0.0),
                          ArgOptParam('fill_mode', 'nearest'),
                          ArgOptParam('cval', 0.0),
                          ArgOptParam('horizontal_flip', False),
                          ArgOptParam('vertical_flip', False),
                          ArgOptParam('rescale', None),
                          ArgOptParam('data_format', None),
                          ArgOptParam('validation_split', 0.0)
                          # not supported as configurable arguments
                          # dtype=None
                          # preprocessing_function=None
                          ]:
            image_generator_args[imgen_arg.name] = default_or_val(imgen_arg, run_cfg)
            params[imgen_arg.name] = default_or_val(imgen_arg, run_cfg)

        # Data preparation
        # https://keras.io/api/preprocessing/image/#imagedatagenerator-class
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
        params.update({key: val for key, val in common_arg.items() if key != 'target_size'})
        for key in ['image_height', 'image_width']:
            params[key] = run_cfg[key]

        train_arg = common_arg.copy()
        train_arg['subset'] = "training"
        train_arg['shuffle'] = True
        train_arg['class_mode'] = 'categorical'
        val_arg = train_arg.copy()
        val_arg['subset'] = "validation"

        if run_cfg['color_mode'] == 'grayscale':
            channels = 1
        else:  # 'rgb'/'rgba'
            channels = 3
        if tf.keras.backend.image_data_format() == 'channels_last':
            input_shape = (run_cfg['image_height'], run_cfg['image_width'], channels)
        else:
            input_shape = (channels, run_cfg['image_height'], run_cfg['image_width'])
        input_tensor = Input(shape=input_shape)

        if 'source' not in app_cfg:
            app_cfg['source'] = 'img'
        if app_cfg['source'] == 'img':
            # ImageDataGenerator
            train_data = train_image_generator.flow_from_dataframe(dataset_df, **train_arg)
            params['train_images'] = len(train_data.filenames)

            val_data = train_image_generator.flow_from_dataframe(dataset_df, **val_arg)
            params['val_images'] = len(val_data.filenames)
        else:
            # Dataset
            train_data = tf.data.Dataset.from_generator(
                lambda: train_image_generator.flow_from_dataframe(dataset_df, **train_arg),
                output_types=(tf.float32, tf.float32),
                output_shapes=([run_cfg['batch_size'], run_cfg['image_height'], run_cfg['image_width'], channels],
                               [run_cfg['batch_size'], run_cfg['class_count']])
            )
            params['train_images'] = floor(len(dataset_df) * (1 - image_generator_args['validation_split']))

            val_data = tf.data.Dataset.from_generator(
                lambda: train_image_generator.flow_from_dataframe(dataset_df, **val_arg),
                output_types=(tf.float32, tf.float32),
                output_shapes=([run_cfg['batch_size'], run_cfg['image_height'], run_cfg['image_width'], channels],
                               [run_cfg['batch_size'], run_cfg['class_count']])
            )
            params['val_images'] = floor(len(dataset_df) * image_generator_args['validation_split'])

        if 'verbose' in app_cfg:
            print(f"{params}")

        misc_args = {}
        for misc in ['gsap_units', 'gsap_activation', 'log_activation', 'run1_optimizer', 'run2_optimizer',
                     'run2_inceptions_to_train', 'run2_train_bn']:
            if misc in run_cfg:
                misc_args[misc] = run_cfg[misc]

        model_args = ModelArgs(run_cfg['device_name'], input_shape, input_tensor, run_cfg['class_count'],
                               train_data, val_data, run_cfg['epochs'], misc_args=misc_args)
        model_args.set_split_total_batch(run_cfg['validation_split'], len(dataset_df), run_cfg['batch_size'])

        params['misc_args'] = misc_args

        # get model function and call it
        method_to_call = getattr(photo_models, run_cfg['function'], None)
        if method_to_call is None:
            error(f"The specified model '{run_cfg['function']} could not be found")
        history = method_to_call(model_args, verbose=('verbose' in app_cfg))

        params['duration'] = f"{duration(start, True):.1f}"

        show_results(history, run_cfg, app_cfg, params)


def show_results(history, run_cfg, app_cfg, params):
    if run_cfg['show_val_loss'] or run_cfg['save_val_loss'] or run_cfg['save_summary']:
        results_path = re.sub(r"<results_path_root>", app_cfg['results_path_root'], run_cfg['results_path'])
        results_path = re.sub(r"<model_name>", run_cfg['name'], results_path)
        timestamp = datetime.datetime.now()
        # std strftime directives
        results_path = re.sub(r"{(.*)}", lambda m: timestamp.strftime(m.group(1)), results_path)
        pathlib.Path(results_path).mkdir(parents=True, exist_ok=True)

        if run_cfg['show_val_loss'] or run_cfg['save_val_loss']:
            acc, val_acc, loss, val_loss = visualise_results(history)

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

                def expand_param(param_dict, sep):
                    param_dict_keys = sorted(param_dict.keys())
                    return sep.join([f"{key2}={str(param_dict[key2])}"
                                     if not isinstance(param_dict[key2], dict)
                                     else expand_param(param_dict[key2], sep[0] * (len(sep) + 1))
                                     for key2 in param_dict_keys])

                keys = sorted(params.keys())
                params_string = f'{",".join([str(params[key]) if not isinstance(params[key], dict) else expand_param(params[key], ";") for key in keys])}'

                new_file = not os.path.exists(filepath)
                with open(filepath, "w" if new_file else "a") as fh:
                    if new_file:
                        line_to_write = f'model,datetime,' \
                                        f'accuracy,val_accuracy,' \
                                        f'loss,val_loss,' \
                                        f'epochs,' \
                                        f'{",".join(keys)},results_folder\n'
                        fh.write(line_to_write)

                    last_result = df.iloc[[-1]]
                    line_to_write = f"{run_cfg['name']},{timestamp.strftime('%Y-%m-%d %H:%M')}," \
                                    f"{last_result['accuracy'].iloc[0]},{last_result['val_accuracy'].iloc[0]}," \
                                    f"{last_result['loss'].iloc[0]},{last_result['val_loss'].iloc[0]}," \
                                    f"{run_cfg['epochs']}," \
                                    f'{params_string},' f'{results_path}\n'
                    fh.write(line_to_write)

        if run_cfg['save_summary']:
            filepath = os.path.join(results_path, "summary.txt")
            print(f"Saving {filepath}")
            with open(filepath, "w") as fh:
                history.model.summary(print_fn=lambda l: fh.write(l + '\n'))

                for i, layer in enumerate(history.model.layers):
                    fh.write(f"{i}\t{layer.name}\n")


def visualise_results(history):
    # Visualize training results
    # loosely based on code from https://www.tensorflow.org/tutorials/images/classification
    epochs_range = range(history.params['epochs'])

    plt.figure(figsize=(8, 8))

    plots = [False, False]
    plots[0] = ('accuracy' in history.history and 'val_accuracy' in history.history)
    plots[1] = ('loss' in history.history and 'val_loss' in history.history)
    ncols = 0
    if plots[0]:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        ncols += 1
    else:
        acc = None
        val_acc = None

    if plots[1]:
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        ncols += 1
    else:
        loss = None
        val_loss = None

    index = 1
    if plots[0]:
        plt.subplot(1, ncols, index)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='best')
        plt.title('Training and Validation Accuracy')
        index += 1

    if plots[1]:
        plt.subplot(1, ncols, index)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='best')
        plt.title('Training and Validation Loss')
        index += 1

    return acc, val_acc, loss, val_loss


def duration(start, verbose):
    seconds = timer() - start
    if verbose:
        print(f"Duration: {seconds:.1f}s")
    return seconds


if __name__ == '__main__':
    main()

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
# def plotImages(images_arr):
#     fig, axes = plt.subplots(1, 5, figsize=(20, 20))
#     axes = axes.flatten()
#     for img, ax in zip(images_arr, axes):
#         ax.imshow(img)
#         ax.axis('off')
#     plt.tight_layout()
#     plt.show()
