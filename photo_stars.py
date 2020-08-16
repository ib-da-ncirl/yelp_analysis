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
import json
import multiprocessing
import os
import pathlib
import re
import sys
from math import floor
from timeit import default_timer as timer
from typing import Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras_preprocessing.image import DataFrameIterator
from sklearn import metrics
from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.models import Model

import photo_models
from misc import less_dangerous_eval, ArgOptParam, default_or_val, pick_device, restrict_gpu_mem, ArgCtrl, predict_img, \
    probability_to_class
from photo_models import ModelArgs
from photo_models.model_args import SplitTotalBatch
from photo_models.model_misc import calc_step_size
from photo_models.tuner import RandomSearchTuner

MIN_PYTHON = (3, 6)
if sys.version_info < MIN_PYTHON:
    sys.exit("Python %s.%s or later is required.\n" % MIN_PYTHON)

CLASS_INDICES_FILENAME = 'class_indices.json'
LABELS_FILENAME = 'labels.txt'
BEST_MODEL_FOLDER = 'best_model'

LAYER_PARAMS = ['dropout_1', 'dropout_2',
                'dense_1', 'dense_2',
                'conv2D_1', 'conv2D_2', 'conv2D_3',
                'pooling_1',
                'log_activation',
                'run1_optimizer', 'run2_optimizer',
                'run1_loss', 'run2_loss',
                'run2_inceptions_to_train', 'run2_train_bn']

DO_EX_DEMO = ['do_training', 'do_prediction', 'do_kerastuning']     # do options excluding demo
APP_LEVEL_CFGS = ['model_path', 'hyper_model_function', 'max_trials', 'executions_per_trial']


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
    arg_ctrl.add_option('k', 'do_kerastuning', 'Do Keras Tuner', typ='flag')
    arg_ctrl.add_option('p', 'do_prediction', 'Do prediction', typ='flag')
    arg_ctrl.add_option('d', 'demo', 'Do demo', has_value=True)
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
        if key not in app_cfg.keys():
            error(f'Missing {key} configuration key')
    # move command line args to where they should be
    for key in ['dataset_path', 'photo_path', 'photo_limit']:
        if key in app_cfg.keys():
            app_cfg['defaults'][key] = app_cfg[key]
            app_cfg.pop(key)
    # required default keys
    for key in ['dataset_path', 'photo_path', 'epochs', 'image_width', 'image_height', 'x_col', 'y_col',
                'color_mode', 'batch_size', 'seed']:
        if key not in app_cfg['defaults'].keys():
            error(f"Missing '{key}' configuration key")
    # check for conflicts

    def all_there(key_list):
        in_args = [a_key for a_key in key_list if a_key in app_cfg.keys() and app_cfg[a_key]]
        return len(in_args) == len(key_list)

    conflicting = ['do_training', 'do_kerastuning']
    if all_there(conflicting):
        error(f"Conflicting configuration keys specified, chose one of {conflicting}")
    # check for ignore
    conflicting = ['do_prediction', 'do_kerastuning']
    if all_there(conflicting):
        warning(f"Ignoring {conflicting[0]} when {conflicting[1]} specified")
        app_cfg[conflicting[0]] = False

    return app_cfg


def error(msg):
    sys.exit(f"Error: {msg}")


def warning(msg):
    print(f"Warning: {msg}")


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


def load_demo(preconfigs: list, demo: str, app_cfg: dict):
    """
    Load model config
    :param preconfigs: List of demos
    :param demo: Name of demo to run
    :param app_cfg: Run config
    :return:
    """
    for config in preconfigs:
        if config['name'] == demo:
            # overwrite app_cfg 'do' options with demo options
            app_cfg.update({key: False for key in DO_EX_DEMO})  # turn off all do
            app_cfg.update({key: val for key, val in config.items() if key in DO_EX_DEMO})  # set do as per demo
            ex_keys = ['name'] + DO_EX_DEMO
            app_cfg['demo'] = {key: val for key, val in config.items() if key not in ex_keys}

            def demo_overwrite(key):
                app_cfg[key] = app_cfg['demo'][key]

            if app_cfg['do_training'] or app_cfg['do_kerastuning']:
                demo_overwrite('run_model')
            elif app_cfg['do_prediction']:
                demo_overwrite('run_model')
                demo_overwrite('model_path')
            break
    else:
        error(f"Unable to find configuration for demo '{demo}")

    return app_cfg


def init_and_load(base_cfg, app_cfg, model_to_run):
    tf.keras.backend.clear_session()  # clear keras state

    run_cfg = base_cfg.copy()

    run_cfg['rescale'] = less_dangerous_eval(run_cfg['rescale'])

    # model config
    run_cfg, hierarchy = load_model_cfg(app_cfg['models'], model_to_run, run_cfg)
    print(f"Loaded model hierarchy: {'->'.join(hierarchy)}")

    if 'demo' in app_cfg.keys():
        # demo mode overwrite relevant keys
        app_cfg.update({key: val for key, val in app_cfg['demo'].items() if key in APP_LEVEL_CFGS})
        run_cfg.update({key: val for key, val in app_cfg['demo'].items() if key not in APP_LEVEL_CFGS})

    return run_cfg


def load_df(app_cfg: dict, run_cfg: dict, mode='train'):
    """
    Load the dataframe
    :param app_cfg:
    :param run_cfg:
    :return:
    """
    print(f"Loading dataframe from '{run_cfg['dataset_path']}'")

    nrows = None
    if isinstance(run_cfg['photo_limit'], str):
        if run_cfg['photo_limit'].lower() != 'none':
            error(f"Unrecognised value for 'photo_limit': '{run_cfg['photo_limit']}'")
    else:
        nrows = run_cfg['photo_limit']

    dataset_df = pd.read_csv(run_cfg['dataset_path'], nrows=nrows, converters={
        'stars_ord': json.loads  # convert string representation of list to a list
    })

    # add columns are ordinal targets i.e. expand list in 'stars_ord' to individual columns
    sample = dataset_df.loc[0, 'stars_ord']
    ord_cols = [f"t{i}" for i in range(len(sample))]
    dataset_df[ord_cols] = dataset_df.apply(lambda row: pd.Series(row['stars_ord']), axis=1, result_type='expand')

    if 'random_batch' in app_cfg:
        sample_ctrl = {
            'n': int(app_cfg['random_batch']) if app_cfg['random_batch'] >= 1.0 else None,
            'frac': app_cfg['random_batch'] if app_cfg['random_batch'] < 1.0 else None
        }
        dataset_df = dataset_df.sample(**sample_ctrl, random_state=1)

    if 'verification_split' in run_cfg and 'validation_split' in run_cfg and mode == 'verify':
        # verification rows are at end of dataframe
        verify_row = int(len(dataset_df) * (1 - run_cfg['verification_split']))
        if verify_row < 1:
            print(f"WARNING: cannot generate verification split {run_cfg['verification_split']}")
        else:
            dataset_df = dataset_df.loc[verify_row:, :]
            print(f"Verification split {len(dataset_df)}")

    return dataset_df, ord_cols


IMAGEDATAGENERATOR_ARGOPTPARAM = [ArgOptParam('featurewise_center', False),
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
                                  ]


def get_image_generator(run_cfg, augmentation: Union[List, bool] = True, verbose=False) \
        -> (ImageDataGenerator, dict, dict, tuple, Input):
    """
    Return an ImageDataGenerator
    :param run_cfg:
    :param augmentation: Include image augmentation settings
    :return:
    """
    image_generator_args = {}
    if isinstance(augmentation, list):
        for imgen_arg in IMAGEDATAGENERATOR_ARGOPTPARAM:
            if imgen_arg.name in augmentation:
                image_generator_args[imgen_arg.name] = default_or_val(imgen_arg, run_cfg)
    elif augmentation:
        for imgen_arg in IMAGEDATAGENERATOR_ARGOPTPARAM:
            image_generator_args[imgen_arg.name] = default_or_val(imgen_arg, run_cfg)

    common_args = {key: run_cfg[key] for key in ['x_col', 'y_col',
                                                 'class_mode', 'color_mode', 'batch_size', 'seed']
                   if key in run_cfg.keys()}
    if common_args['class_mode'] == 'raw' and 'ord_cols' in run_cfg.keys():
        common_args['y_col'] = run_cfg['ord_cols']
    common_args['directory'] = run_cfg['photo_path']
    common_args['target_size'] = (run_cfg['image_height'], run_cfg['image_width'])

    if run_cfg['color_mode'] == 'grayscale':
        channels = 1
    else:  # 'rgb'/'rgba'
        channels = 3
    if tf.keras.backend.image_data_format() == 'channels_last':
        input_shape = (run_cfg['image_height'], run_cfg['image_width'], channels)
    else:
        input_shape = (channels, run_cfg['image_height'], run_cfg['image_width'])
    input_tensor = Input(shape=input_shape)

    if verbose:
        print(f"get_image_generator: generator_args {image_generator_args}")
        print(f"get_image_generator: common_args {common_args}")

    # https://keras.io/api/preprocessing/image/#imagedatagenerator-class
    return ImageDataGenerator(**image_generator_args), image_generator_args, common_args, input_shape, input_tensor


def channels_from_shape(input_shape):
    """ Number of channels from input shape tuple """
    return input_shape[2] if tf.keras.backend.image_data_format() == 'channels_last' else input_shape[0]


def flow_from_df(source: str, image_generator: ImageDataGenerator, df: pd.DataFrame, args: dict, run_cfg: dict,
                 input_shape: tuple, verbose=False) -> (Union[DataFrameIterator, tf.data.Dataset], int):
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
    if verbose:
        print(f"flow_from_df: args {args}")

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
        if 'validation_split' in args.keys():
            count = floor(len(df) * (1 - args['validation_split']))
        else:
            count = len(df)
        img_data_flow.prefetch(run_cfg['batch_size'])

    return img_data_flow, count


def tune_model(model_args: ModelArgs, run_cfg: dict, app_cfg: dict):

    # get model function and call it
    method_to_call = getattr(photo_models, app_cfg['hyper_model_function'], None)
    if method_to_call is None:
        error(f"The specified model '{app_cfg['hyper_model_function']} could not be found")

    with tf.device(model_args.device_name):
        print(f"Using '{model_args.device_name}'")

        tuner = RandomSearchTuner(
                method_to_call(model_args),
                'val_accuracy',
                app_cfg['max_trials'],
                model_args,
                executions_per_trial=app_cfg['executions_per_trial'],
                directory=get_storage_path(run_cfg, app_cfg, app_cfg['tuning_directory']))

        tuner.search_space_summary()
        tuner.search()
        tuner.results_summary()


def do_train(app_cfg: dict, base_cfg: dict, start: float, timestamp: datetime):
    """
    Do training
    :param app_cfg: Application config
    :param base_cfg: Models base configuration
    :param start:
    :param timestamp:
    :return:
    """

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
        print(f"\nRunning {run_cfg['desc']} ({model_idx}/{len(models_run_list)})")
        if app_cfg['verbose']:
            print(f"Run config: {run_cfg}")

        params = {'dataset_path': run_cfg['dataset_path']}

        # load dataset
        dataset_df, ord_cols = load_df(app_cfg, run_cfg)

        run_cfg['ord_cols'] = ord_cols

        if 'verification_split' in run_cfg:
            split = int(len(dataset_df) - (len(dataset_df) * run_cfg['verification_split']))
            train_val_df = dataset_df.loc[:split - 1, :]
            verify_df = dataset_df.loc[split:, :]
        else:
            train_val_df = dataset_df
            verify_df = None

        split = int(len(train_val_df) - (len(train_val_df) * run_cfg['validation_split']))
        train_freq = pd.crosstab(train_val_df.loc[:split - 1, :]['stars'], columns='count')
        train_freq['percent'] = train_freq['count'] / train_freq['count'].sum()
        val_freq = pd.crosstab(train_val_df.loc[split:, :]['stars'], columns='count')
        val_freq['percent'] = val_freq['count'] / val_freq['count'].sum()
        total_freq = pd.crosstab(train_val_df['stars'], columns='count')
        total_freq['percent'] = total_freq['count'] / total_freq['count'].sum()
        print("Training data")
        print(train_freq)
        print("Validation data")
        print(val_freq)
        print("Training-Validation validation")
        print(total_freq)

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
        val_arg = train_arg.copy()
        val_arg['subset'] = "validation"

        if 'source' not in app_cfg:
            app_cfg['source'] = 'img'

        train_data, params['train_images'] = flow_from_df(app_cfg['source'], train_image_generator, train_val_df,
                                                          train_arg, run_cfg, input_shape)
        val_data, params['val_images'] = flow_from_df(app_cfg['source'], train_image_generator, train_val_df,
                                                      val_arg, run_cfg, input_shape)

        if train_data.class_mode == 'categorical':
            # categorical mode
            if train_data.class_indices != val_data.class_indices:
                raise ValueError(f"Indices mismatch: train={train_data.class_indices}, "
                                 f"validation={val_data.class_indices}")
            # swap key/val i.e. index becomes key and category becomes value
            params['train_class_indices'] = {val: key for key, val in train_data.class_indices.items()}
            params['val_class_indices'] = {val: key for key, val in val_data.class_indices.items()}
            model_classes = params['train_class_indices']
        elif train_data.class_mode == 'raw':
            model_classes = None
        else:
            raise NotImplementedError(f"Class mode '{train_data.class_mode}' not implemented")

        misc_args = {key: val for key, val in run_cfg.items() if key in LAYER_PARAMS}

        model_args = ModelArgs(run_cfg['device_name'], input_shape, input_tensor, run_cfg['class_count'],
                               train_data, val_data, run_cfg['epochs'], misc_args=misc_args)
        model_args.set_split_total_batch(run_cfg['validation_split'], len(train_val_df), run_cfg['batch_size'])
        model_args.callbacks = ModelCheckpoint(monitor='val_loss', save_best_only=True,
                                               filepath=os.path.join(get_results_path(run_cfg, app_cfg,
                                                                                      timestamp=timestamp),
                                                                     BEST_MODEL_FOLDER)
                                               )

        params['misc_args'] = misc_args

        if app_cfg['verbose']:
            print(f"{params}")

        if app_cfg['do_kerastuning']:
            tune_model(model_args, run_cfg, app_cfg)

        else:
            # get model function and call it
            method_to_call = getattr(photo_models, run_cfg['function'], None)
            if method_to_call is None:
                error(f"The specified model '{run_cfg['function']} could not be found")
            history = method_to_call(model_args, verbose=app_cfg['verbose'])

            params['duration'] = f"{duration(start, True):.1f}"

            if app_cfg['do_prediction'] and verify_df is not None:
                do_predict(app_cfg, base_cfg, model=history.model, ord_cols=ord_cols, dataset_df=verify_df,
                           class_mode=train_data.class_mode, model_classes=model_classes)

            show_results(history, run_cfg, app_cfg, params, timestamp=timestamp)


def do_predict(app_cfg: dict, base_cfg: dict, model: Model = None, ord_cols: list = None,
               dataset_df: pd.DataFrame = None, class_mode=None, model_classes=None, start: float = None,
               timestamp: datetime = None,
               batch_mode: bool = True):
    """
    Do prediction
    :param app_cfg: Application config
    :param base_cfg: Models base configuration
    :return:
    """
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
            model_path = get_storage_path(run_cfg, app_cfg, app_cfg['model_path'])
            print(f"Loading model from {model_path}")
            model = keras.models.load_model(model_path)

            if class_mode is None:
                # TODO best_model's need class indices from normal model, need to detect this
                filepath = os.path.join(app_cfg['model_path'], CLASS_INDICES_FILENAME)
                if os.path.exists(filepath):
                    class_mode = 'categorical'
                else:
                    class_mode = 'raw'

                if class_mode == 'categorical':
                    print(f"Reading class indices from {filepath}")
                    with open(filepath) as fhin:
                        model_classes = json.load(fhin)
                    # need to convert keys to int, as they were originally indices, and
                    # convert the category name 'x_y' to its star value x.y
                    model_classes = {int(key): {'cat': val,
                                                'stars': float(val.replace('_', '.'))
                                                } for key, val in model_classes.items()}

        if dataset_df is None:
            # load dataset
            dataset_df, ord_cols = load_df(app_cfg, base_cfg, mode='verify')

        run_cfg['ord_cols'] = ord_cols

        # Data preparation
        run_cfg['shuffle'] = False
        if class_mode != 'categorical':
            run_cfg['class_mode'] = None  # the generator will only yield batches of image data

        run_cfg.pop('ord_cols')

        image_generator, _, common_args, input_shape, input_tensor = \
            get_image_generator(run_cfg, augmentation=['rescale'], verbose=True)

        train_arg = common_args.copy()

        if 'source' not in app_cfg:
            app_cfg['source'] = 'img'

        train_data, num_train_images = flow_from_df(app_cfg['source'], image_generator, dataset_df,
                                                    train_arg, run_cfg, input_shape, verbose=True)

        # train_data.reset()

        processed_cnt = 0
        success_cnt = 0
        if batch_mode:
            predictions = model.predict(train_data, verbose=1,
                                        steps=calc_step_size(train_data,
                                                             SplitTotalBatch(0.0, len(dataset_df),
                                                                             run_cfg['batch_size']),
                                                             None)
                                        )

            results = probability_to_class(predictions, classes=model_classes)

            actual_half_stars = (dataset_df['stars'] * 2).astype(int).values

            ver_freq = pd.crosstab(dataset_df['stars'], columns='count')
            ver_freq['percent'] = ver_freq['count'] / ver_freq['count'].sum()
            print("Prediction data")
            print(ver_freq)

            if class_mode == 'categorical':
                predict_half_stars = np.array([int(pred[0].class_spec['stars'] * 2) for pred in results], dtype=int)
            else:
                predict_half_stars = np.sum(np.round(results).astype(int), axis=1) + 1  # +1 as starts at 1 star TODO use min

            star_list = [f"{val['stars']:.1f}" for val in model_classes.values()]
            print(f"Model star range: {', '.join(star_list)}")

            if 'confusion_matrix' in run_cfg.keys():
                kwargs = run_cfg['confusion_matrix'].copy()
                if 'normalize' in kwargs and kwargs['normalize'] == 'none':
                    kwargs['normalize'] = None


            print(f"Verification sample: min star {dataset_df['stars'].min()}, max star {dataset_df['stars'].max()}")
            print(f"Predictions: min star {predict_half_stars.min()/2}, max star {predict_half_stars.max()/2}")

            print("Confusion Matrix")
            star_list = [f'{x/2:.1f}' for x in range(predict_half_stars.min(), actual_half_stars.max() + 1)]
            print(f"Stars: {', '.join(star_list)}")
            print(metrics.confusion_matrix(actual_half_stars, predict_half_stars, **kwargs))

            print("Accuracy Score")
            print(metrics.accuracy_score(actual_half_stars, predict_half_stars))

            print('Classification Report')
            print(metrics.classification_report(actual_half_stars, predict_half_stars))

        else:
            # A DataFrameIterator yields tuples of (x, y) where x is a numpy array containing a batch of images with
            # shape (batch_size, *target_size, channels) and y is a numpy array of corresponding labels.
            success_cnt = 0
            while processed_cnt < num_train_images:
                # load next image
                if train_arg['class_mode'] is None:
                    images = next(train_data)
                else:
                    images, labels = next(train_data)

                for i in range(len(images)):

                    row = dataset_df.iloc[processed_cnt, :]
                    if row['photo_file'] != train_data.filenames[processed_cnt]:
                        raise ValueError(f"Unexpected file: df {row['photo_file']}  "
                                         f"data {train_data.filenames[processed_cnt]}")

                    results = predict_img(images[i], model, model_classes)
                    for result in results:
                        print(f"{processed_cnt:4d}: {row['photo_id']} - actual {row['stars']}")
                        if class_mode == 'categorical':
                            line = ''
                            for pred in result:
                                match = (row['stars'] == pred.class_spec['stars'])
                                line = f"    > predicted {pred.class_spec['cat']}  probability {pred.probability} " \
                                       f"{'Y' if match else '-'}\n"
                                if match:
                                    success_cnt += 1
                        else:
                            rounded = [int(round(x)) for x in result]
                            line = f"    > actual {row['stars_ord']}  predicted {result}\n" \
                                   f"    >        {rounded}"
                            if row['stars_ord'] == rounded:
                                success_cnt += 1
                        print(line)

                    processed_cnt += 1

            print(f"Correctly predicted {success_cnt} of {num_train_images} ({float(success_cnt) / num_train_images:.4f})")

    if start is not None:
        duration(start, True)


def main_run(app_cfg: dict):
    start = timer()
    run_timestamp = datetime.datetime.now()

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
    print(f"Selected '{base_cfg['device_name']}'")

    if app_cfg['do_training'] or app_cfg['do_kerastuning']:
        if 'run_model' in app_cfg:
            do_train(app_cfg, base_cfg, start, run_timestamp)
    elif app_cfg['do_prediction']:
        if 'model_path' in app_cfg:
            do_predict(app_cfg, base_cfg, start=start, timestamp=run_timestamp)


def main():
    # load app config
    app_cfg = get_app_config(sys.argv[0], sys.argv[1:])

    is_demo = 'demo' in app_cfg.keys()
    if is_demo:
        app_cfg = load_demo(app_cfg['preconfigured'], app_cfg['demo'], app_cfg)

        if 'msg' in app_cfg['demo'].keys():
            print("*" * len(app_cfg['demo']['msg']))
            print(app_cfg['demo']['msg'])
            print("*" * len(app_cfg['demo']['msg']))
            if 'proceed' in app_cfg['demo'].keys():
                if app_cfg['demo']['proceed'] == 'query':
                    proceed = ""
                    while len(proceed) == 0:
                        proceed = input("Do you wish to proceed [y/n]: ")
                        if len(proceed):
                            proceed = proceed.lower()
                            if proceed != 'y' and proceed != 'n':
                                proceed = ""
                    if proceed == 'n':
                        exit(0)

    # sometimes tensorflow doesn't properly release gpu memory, and you need to reboot, (or switch to cpu)
    # so thanks to https://github.com/tensorflow/tensorflow/issues/36465#issuecomment-582749350
    process_eval = multiprocessing.Process(target=main_run, args=(app_cfg,))
    process_eval.start()
    process_eval.join()


def get_storage_path(run_cfg: dict, app_cfg: dict, path: str, timestamp=None):
    store_path = re.sub(r"<results_path_root>", app_cfg['results_path_root'], path)
    store_path = re.sub(r"<model_name>", run_cfg['name'], store_path)
    store_path = re.sub(r"<hyper_model_function>", app_cfg['hyper_model_function'], store_path)
    if timestamp is None:
        timestamp = datetime.datetime.now()
    # std strftime directives
    store_path = re.sub(r"{(.*)}", lambda m: timestamp.strftime(m.group(1)), store_path)
    return store_path


def get_results_path(run_cfg: dict, app_cfg: dict, timestamp: datetime = None, create: bool = True):
    results_path = get_storage_path(run_cfg, app_cfg, run_cfg['results_path'], timestamp=timestamp)
    if create:
        pathlib.Path(results_path).mkdir(parents=True, exist_ok=True)
    return results_path


def show_results(history, run_cfg: dict, app_cfg: dict, params: dict, timestamp: datetime = None):
    if run_cfg['show_val_loss'] or run_cfg['save_val_loss'] or run_cfg['save_summary'] or run_cfg['save_model']:
        results_path = get_results_path(run_cfg, app_cfg, timestamp=timestamp)

        if run_cfg['show_val_loss'] or run_cfg['save_val_loss']:
            acc, val_acc, loss, val_loss = visualise_results(history)

            if run_cfg['show_val_loss']:
                # display val/loss graph
                plt.show()
            if run_cfg['save_val_loss']:
                # save val/loss graph image
                filepath = os.path.join(results_path, f"{run_cfg['name']}.png")
                print(f"Saving '{filepath}'")
                plt.savefig(filepath)

                # save val/loss data
                df = pd.DataFrame(data={
                    'accuracy': acc,
                    'val_accuracy': val_acc,
                    'loss': loss,
                    'val_loss': val_loss
                })
                filepath = os.path.join(results_path, f"{run_cfg['name']}.csv")
                print(f"Saving '{filepath}'")
                df.to_csv(filepath)

                # save val/loss data to overall results
                filepath = os.path.join(app_cfg['results_path_root'], f"result_log.csv")
                print(f"Updating '{filepath}'")

                def expand_param(param_dict, key, sep, level):
                    param_obj = param_dict[key]
                    if isinstance(param_obj, list):
                        line = sep.join(param_obj)
                    else:
                        param_dict_keys = sorted(param_obj.keys())
                        line = sep.join([f"{key2}={str(param_obj[key2])}"
                                         if not isinstance(param_obj[key2], (list, dict))
                                         else expand_param(param_obj, key2, sep[0] * (len(sep) + 1), level + 1)
                                         for key2 in param_dict_keys])
                    return line if level == 0 else f" {key}={line}"

                keys = sorted(params.keys())
                params_string = f'{",".join([str(params[key]) if not isinstance(params[key], (list, dict)) else expand_param(params, key, ";", 0) for key in keys])}'

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
            print(f"Saving model summary to '{filepath}'")
            with open(filepath, "w") as fh:
                history.model.summary(print_fn=lambda l: fh.write(l + '\n'))

                for i, layer in enumerate(history.model.layers):
                    fh.write(f"{i}\t{layer.name}\n")

            filepath = os.path.join(results_path, "model.png")
            print(f"Saving model plot to '{filepath}'")
            keras.utils.plot_model(history.model, show_shapes=True, to_file=filepath)

        if run_cfg['save_model']:
            filepath = os.path.join(results_path, history.model.name)
            print(f"Saving model to '{filepath}'")
            history.model.save(filepath)

            if 'val_class_indices' in params:
                filepath = os.path.join(results_path, history.model.name, CLASS_INDICES_FILENAME)
                print(f"Saving class indices to '{filepath}'")
                with open(filepath, 'w') as fhout:
                    json.dump(params['val_class_indices'], fhout)
            elif 'labels' in params:
                filepath = os.path.join(results_path, history.model.name, LABELS_FILENAME)
                print(f"Saving labels to '{filepath}'")
                with open(filepath, 'w') as fhout:
                    fhout.write(f"{str(params['labels'])}\n")


def visualise_results(history):
    # Visualize training results
    # loosely based on code from https://www.tensorflow.org/tutorials/images/classification
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
        epochs_range = range(len(acc))
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='best')
        plt.title('Training and Validation Accuracy')
        index += 1

    if plots[1]:
        plt.subplot(1, ncols, index)
        epochs_range = range(len(loss))
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
