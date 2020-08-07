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
import json
from math import floor
from timeit import default_timer as timer

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Input
from tensorflow.python.keras.models import Model

import photo_models
from misc import less_dangerous_eval, ArgOptParam, default_or_val, pick_device, restrict_gpu_mem, ArgCtrl, predict, \
    predict_img
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
    arg_ctrl.add_option('m', 'modelling_device', 'TensorFlow preferred modelling device; e.g. /cpu:0', has_value=True)
    arg_ctrl.add_option('r', 'run_model', 'Model to run', has_value=True)
    arg_ctrl.add_option('x', 'execute_model', 'Model to load and execute', has_value=True)
    arg_ctrl.add_option('t', 'do_training', 'Do model training', typ='flag')
    arg_ctrl.add_option('p', 'do_prediction', 'Do prediction', typ='flag')
    arg_ctrl.add_option('s', 'source', "Model source; 'img' = ImageDataGenerator or 'ds' = Dataset", has_value=True)
    arg_ctrl.add_option('b', 'random_batch', "If < 1, percent of available photo to randomly sample, "
                                             "else number to randomly sample",
                        has_value=True, typ=float)
    arg_ctrl.add_option('l', 'photo_limit', "Max number of photos to use; 'none' to use all available, or a number",
                        has_value=True, typ=int)
    arg_ctrl.add_option('v', 'verbose', 'Verbose mode', typ='flag')

    app_cfg = arg_ctrl.get_app_config(args)

    # check some basic configs exist
    for key in ['defaults', 'run_model']:  # required root level keys
        if key not in app_cfg:
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

    return run_cfg, hierarchy


def init_and_load(base_cfg, app_cfg, model_to_run):
    tf.keras.backend.clear_session()  # clear keras state

    run_cfg = base_cfg.copy()

    run_cfg['rescale'] = less_dangerous_eval(run_cfg['rescale'])

    # model config
    run_cfg, hierarchy = load_model_cfg(app_cfg['models'], model_to_run, run_cfg)
    print(f"Loaded model hierarchy: {'->'.join(hierarchy)}")

    return run_cfg


def load_df(app_cfg, run_cfg):
    """
    Load the dataframe
    :param app_cfg:
    :param run_cfg:
    :return:
    """
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

    return dataset_df


def get_image_generator(run_cfg, augmentation=True):
    """
    Return an ImageDataGenerator
    :param run_cfg:
    :param augmentation: Include image augmentation settings
    :return:
    """
    image_generator_args = {}
    if augmentation:
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

    common_args = {
        'directory': run_cfg['photo_path'],
        'x_col': run_cfg['x_col'],
        'y_col': run_cfg['y_col'],
        'color_mode': run_cfg['color_mode'],
        'batch_size': run_cfg['batch_size'],
        'seed': run_cfg['seed'],
        'target_size': (run_cfg['image_height'], run_cfg['image_width']),
    }

    if run_cfg['color_mode'] == 'grayscale':
        channels = 1
    else:  # 'rgb'/'rgba'
        channels = 3
    if tf.keras.backend.image_data_format() == 'channels_last':
        input_shape = (run_cfg['image_height'], run_cfg['image_width'], channels)
    else:
        input_shape = (channels, run_cfg['image_height'], run_cfg['image_width'])
    input_tensor = Input(shape=input_shape)

    # https://keras.io/api/preprocessing/image/#imagedatagenerator-class
    return ImageDataGenerator(**image_generator_args), image_generator_args, common_args, input_shape, input_tensor


def channels_from_shape(input_shape):
    """ Number of channels from input shape tuple """
    return input_shape[2] if tf.keras.backend.image_data_format() == 'channels_last' else input_shape[0]


def flow_from_df(source, image_generator, df, args, run_cfg, input_shape):
    """
    Create an ImageDataGenerator.flow_from_dataframe
    :param source:
    :param image_generator:
    :param df:
    :param args:
    :param run_cfg:
    :param input_shape:
    :return:
    """
    if source == 'img':
        # ImageDataGenerator
        img_data_flow = image_generator.flow_from_dataframe(df, **args)
        count = len(img_data_flow.filenames)
    else:
        # Dataset
        img_data_flow = tf.data.Dataset.from_generator(
            lambda: image_generator.flow_from_dataframe(df, **args),
            output_types=(tf.float32, tf.float32),
            output_shapes=([run_cfg['batch_size'], run_cfg['image_height'], run_cfg['image_width'],
                            channels_from_shape(input_shape)],
                           [run_cfg['batch_size'], run_cfg['class_count']])
        )
        count = floor(len(df) * (1 - args['validation_split']))

    return img_data_flow, count


def do_train(app_cfg, base_cfg, start):

    # model list
    if isinstance(app_cfg['run_model'], str):
        models_run_list = [app_cfg['run_model']]
    else:
        models_run_list = app_cfg['run_model']

    model_idx = 0
    for model_to_run in models_run_list:

        # model config
        run_cfg = init_and_load(base_cfg, app_cfg, model_to_run)

        # required keys
        for key in ['function']:
            if key not in run_cfg.keys():
                error(f"Missing '{key}' configuration key")

        model_idx += 1
        print(f"\nRunning {run_cfg['desc']} ({model_idx/len(models_run_list)})")
        if app_cfg['verbose']:
            print(f"Run config: {run_cfg}")

        params = {'dataset_path': run_cfg['dataset_path']}

        # load dataset
        dataset_df = load_df(app_cfg, run_cfg)

        if 'verification_split' in run_cfg:
            split = int(len(dataset_df) - (len(dataset_df) * run_cfg['verification_split']))
            train_val_df = dataset_df.loc[:split-1, :]
            verify_df = dataset_df.loc[split:, :]
        else:
            train_val_df = dataset_df
            verify_df = None

        # Data preparation
        # https://keras.io/api/preprocessing/image/#imagedatagenerator-class
        train_image_generator, image_generator_args, common_args, input_shape, input_tensor = \
            get_image_generator(run_cfg)

        params.update(image_generator_args)
        params.update(common_args)
        # replace target size tuple with height/width
        params.pop('target_size')
        for key in ['image_height', 'image_width']:
            params[key] = run_cfg[key]

        train_arg = common_args.copy()
        train_arg['subset'] = "training"
        train_arg['shuffle'] = True
        train_arg['class_mode'] = 'categorical'
        val_arg = train_arg.copy()
        val_arg['subset'] = "validation"

        if 'source' not in app_cfg:
            app_cfg['source'] = 'img'

        train_data, params['train_images'] = flow_from_df(app_cfg['source'], train_image_generator, train_val_df,
                                                          train_arg, run_cfg, input_shape)
        val_data, params['val_images'] = flow_from_df(app_cfg['source'], train_image_generator, train_val_df,
                                                      val_arg, run_cfg, input_shape)

        if train_data.class_indices != val_data.class_indices:
            raise ValueError(f"Indices mismatch: train={train_data.class_indices}, validation={val_data.class_indices}")
        # swap key/val i.e. index becomes key and category becomes value
        params['train_class_indices'] = {val: key for key, val in train_data.class_indices.items()}
        params['val_class_indices'] = {val: key for key, val in val_data.class_indices.items()}

        misc_args = {}
        for misc in ['gsap_units', 'gsap_activation', 'log_activation', 'run1_optimizer', 'run2_optimizer',
                     'run2_inceptions_to_train', 'run2_train_bn']:
            if misc in run_cfg:
                misc_args[misc] = run_cfg[misc]

        model_args = ModelArgs(run_cfg['device_name'], input_shape, input_tensor, run_cfg['class_count'],
                               train_data, val_data, run_cfg['epochs'], misc_args=misc_args)
        model_args.set_split_total_batch(run_cfg['validation_split'], len(train_val_df), run_cfg['batch_size'])

        params['misc_args'] = misc_args

        if app_cfg['verbose']:
            print(f"{params}")

        # get model function and call it
        method_to_call = getattr(photo_models, run_cfg['function'], None)
        if method_to_call is None:
            error(f"The specified model '{run_cfg['function']} could not be found")
        history = method_to_call(model_args, verbose=app_cfg['verbose'])

        params['duration'] = f"{duration(start, True):.1f}"

        show_results(history, run_cfg, app_cfg, params)

        if verify_df is not None:
            do_predict(app_cfg, base_cfg, start, model=history.model, dataset_df=verify_df,
                       class_indices=params['val_class_indices'])


def do_predict(app_cfg, base_cfg, start, model: Model = None, dataset_df=None, class_indices=None):

    # model list
    if isinstance(app_cfg['run_model'], str):
        models_run_list = [app_cfg['run_model']]
    else:
        models_run_list = app_cfg['run_model']

    for model_to_run in models_run_list:
        # model config
        run_cfg = init_and_load(base_cfg, app_cfg, model_to_run)

        if model is None:
            # load model weights
            print(f"Loading model from {app_cfg['model_path']}")
            model = keras.models.load_model(app_cfg['model_path'])

            filepath = os.path.join(app_cfg['model_path'], 'class_indices.json')
            print(f"Reading class indices from {filepath}")
            with open(filepath) as json_file:
                class_indices = json.load(json_file)
            # need to convert keys to int, as they were originally indices
            class_indices = {int(key): val for key, val in class_indices.items()}

        if dataset_df is None:
            # load dataset
            dataset_df = load_df(app_cfg, base_cfg)

        # Data preparation
        image_generator, _, common_args, input_shape, input_tensor = \
            get_image_generator(run_cfg, augmentation=False)

        train_arg = common_args.copy()
        train_arg['subset'] = "training"
        train_arg['shuffle'] = True
        train_arg['class_mode'] = 'categorical'

        if 'source' not in app_cfg:
            app_cfg['source'] = 'img'

        train_data, num_train_images = flow_from_df(app_cfg['source'], image_generator, dataset_df,
                                                    train_arg, run_cfg, input_shape)

        # A DataFrameIterator yields tuples of (x, y) where x is a numpy array containing a batch of images with
        # shape (batch_size, *target_size, channels) and y is a numpy array of corresponding labels.
        processed_cnt = 0
        success_cnt = 0
        while processed_cnt < num_train_images:
            # load next batch of images
            images, labels = next(train_data)

            for i in range(len(images)):

                row = dataset_df.iloc[processed_cnt, :]
                if row['photo_file'] != train_data.filenames[processed_cnt]:
                    raise ValueError(f"Not expected file: df {row['photo_file']}  data {train_data.filenames[processed_cnt]}")

                results = predict_img(images[i], model, class_indices)
                for result in results:
                    print(f"{processed_cnt:4d}: {row['photo_id']} - actual {row['stars']}")
                    for pred in result:
                        print(f"    > predicted {pred[0]}  probability {pred[1]} => "
                              f"{'Y' if row['stars'] == pred[0] else '-'}")
                        if row['stars'] == pred[0]:
                            success_cnt += 1

                processed_cnt += 1

        print(f"Correctly predicted {success_cnt} of {num_train_images} ({float(success_cnt)/num_train_images*100:.1f}%)")


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

    # device selection
    base_cfg['device_name'], fallback = pick_device(app_cfg['modelling_device'])
    if fallback:
        print(f"Device '{app_cfg['modelling_device']}' not available")
    print(f"Using '{base_cfg['device_name']}'")

    if app_cfg['do_training']:
        if 'run_model' in app_cfg:
            do_train(app_cfg, base_cfg, start)
    elif app_cfg['do_prediction']:
        if 'model_path' in app_cfg:
            do_predict(app_cfg, base_cfg, start)


def show_results(history, run_cfg, app_cfg, params):
    if run_cfg['show_val_loss'] or run_cfg['save_val_loss'] or run_cfg['save_summary'] or run_cfg['save_model']:
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
                                        f'{",".join(keys)},results_folder,model_saved\n'
                        fh.write(line_to_write)

                    last_result = df.iloc[[-1]]
                    line_to_write = f"{run_cfg['name']},{timestamp.strftime('%Y-%m-%d %H:%M')}," \
                                    f"{last_result['accuracy'].iloc[0]},{last_result['val_accuracy'].iloc[0]}," \
                                    f"{last_result['loss'].iloc[0]},{last_result['val_loss'].iloc[0]}," \
                                    f"{run_cfg['epochs']}," \
                                    f"{params_string},{results_path}," \
                                    f"{'yes' if 'save_model' in run_cfg and run_cfg['save_model'] else 'no'}\n"
                    fh.write(line_to_write)

        if run_cfg['save_summary']:
            filepath = os.path.join(results_path, "summary.txt")
            print(f"Saving {filepath}")
            with open(filepath, "w") as fh:
                history.model.summary(print_fn=lambda l: fh.write(l + '\n'))

                for i, layer in enumerate(history.model.layers):
                    fh.write(f"{i}\t{layer.name}\n")

        if run_cfg['save_model']:
            filepath = os.path.join(results_path, history.model.name)
            print(f"Saving model to {filepath}")
            history.model.save(filepath)

            filepath = os.path.join(results_path, history.model.name, 'class_indices.json')
            print(f"Saving class indices to {filepath}")
            with open(filepath, 'w') as fhout:
                json.dump(params['val_class_indices'], fhout)


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
