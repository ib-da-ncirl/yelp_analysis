# The MIT License (MIT)
# Copyright (c) 2019 Ian Buttimer

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from io import SEEK_SET
from os import (
    path,
    getcwd
)
import re
import logging
import yaml
import pkg_resources
from .get_env import test_file_path


def load_cfg_file(cfg_file, keys, separator='='):
    """
    Read settings from specified configuration file
    :param cfg_file: Configuration file descriptor to load
    :param keys: List of keys for which to retrieve values
    :param separator: Optional key/value separator,defaults to '='
    :return: dict of key/values
    :rtype: dict
    :raises ValueError when invalid configuration entries detected
    """
    if cfg_file is None:
        raise ValueError('Missing configuration file argument')

    config = {}
    cfg_file.seek(0, SEEK_SET)  # seek start of file
    count = 0
    for line in cfg_file:
        line = line.strip()
        count += 1

        # skip blank or commented lines
        if len(line) == 0:
            continue
        if line.startswith('#'):
            continue

        key_val = re.match(rf'(\w+)\s*{separator}(.*)', line)
        if not key_val:
            raise ValueError(f'Invalid configuration file entry on line {count}: {line}')

        key = key_val.groups()[0].lower().strip()
        if len(key) == 0:
            raise ValueError(f'Missing key entry on line {count}: {line}')
        value = key_val.groups()[1].strip()
        if len(value) == 0:
            raise ValueError(f'Missing value entry on line {count}: {line}')

        if key in keys:
            config[key] = value
        else:
            logging.info(f'Ignoring unknown entry on line {count}')

    return config


def load_cfg_filename(cfg_filename, keys, separator='='):
    """
    Read settings from specified configuration file
    :param cfg_filename: Path of configuration file to load
    :param keys: List of keys for which to retrieve values
    :param separator: Optional key/value separator,defaults to '='
    :return: dict of key/values
    :rtype: dict
    :raises ValueError when invalid configuration entries detected
    """
    if cfg_filename is None:
        raise ValueError('Missing configuration file argument')
    if not isinstance(cfg_filename, str):
        raise ValueError('Invalid configuration file argument: expected string')
    if not path.exists(cfg_filename):
        raise ValueError(f'Configuration file does not exist: {cfg_filename}\n'
                         f'  Current working directory: {getcwd()}')

    with open(cfg_filename, 'r') as cfg_file:
        config = load_cfg_file(cfg_file, keys, separator=separator)

    return config


def load_yaml(yaml_path, key=None):
    """
    Load yaml file and return the configuration dictionary
    :param yaml_path: path to the yaml configuration file
    :param key: configuration key to return; default is all keys
    :return: configuration dictionary
    :rtype: dict
    """
    # verify path
    if not path.exists(yaml_path):
        raise ValueError(f'Invalid path: {yaml_path}')
    if not test_file_path(yaml_path):
        raise ValueError(f'Not a file path: {yaml_path}')

    config_dict = None
    with open(rf'{yaml_path}') as file:
        # The FullLoader parameter handles the conversion from YAML scalar values to Python the dictionary format
        if pkg_resources.get_distribution("PyYAML").version.startswith('5'):
            # https://github.com/yaml/pyyaml/wiki/PyYAML-yaml.load(input)-Deprecation
            configs = yaml.load(file, Loader=yaml.FullLoader)
        else:
            configs = yaml.load(file)
        if key is not None:
            config_dict = configs[key]
        else:
            config_dict = configs

    return config_dict
