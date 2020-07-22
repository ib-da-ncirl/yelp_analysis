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

import os

__ISFILE = 0
__ISDIR = 1


def __get_path(environ, chk_type, req_file_desc):
    """
    Get a file path, either from an environment variable or input
    :param environ: Environment variable to
    :type chk_type: type to check for; __ISFILE or __ISDIR
    :param req_file_desc: Description of required file
    :return: file path or None
    """
    path = None
    approve = 'n'
    if isinstance(environ, str):
        path = os.environ.get(environ)
        if chk_type == __ISFILE:
            if test_file_path(path,
                              nonexistent=f'>> The path specified in the environment variable {environ} '
                                          f'does not exist, ignoring',
                              not_ok_msg=f'>> The path specified in the environment variable {environ} '
                                         f'is not a file, ignoring'):
                approve = 'y'
        elif chk_type == __ISDIR:
            if test_dir_path(path,
                             nonexistent=f'>> The path specified in the environment variable {environ} '
                                         f'does not exist, ignoring',
                             not_ok_msg=f'>> The path specified in the environment variable {environ} '
                                         f'is not a directory, ignoring'):
                approve = 'y'
        else:
            raise ValueError(f'Unknown check type {chk_type}')

    if approve != 'q':
        print(f"Current directory is: {os.getcwd()}")
        while path is None:
            path = input(f"Enter path to {req_file_desc} [or 'q' to quit]: ")
            if path.lower() == 'q':
                path = None
                break

            if chk_type == __ISFILE:
                if not test_file_path(path,
                                      nonexistent=f'>> The path does not exist',
                                      not_ok_msg=f'>> Not a file'):
                    path = None
            elif chk_type == __ISDIR:
                if not test_dir_path(path,
                                     nonexistent=f'>> The path does not exist',
                                     not_ok_msg=f'>> Not a directory'):
                    path = None

    return path


def get_file_path(environ, req_file_desc):
    """
    Get a file path, either from an environment variable or input
    :param environ: Environment variable to read
    :param req_file_desc: Description of required file
    :return: file path or None
    """
    return __get_path(environ, __ISFILE, req_file_desc)


def get_dir_path(environ, req_file_desc):
    """
    Get a directory path, either from an environment variable or input
    :param environ: Environment variable to read
    :param req_file_desc: Description of required directory
    :return: dir path or None
    """
    return __get_path(environ, __ISDIR, req_file_desc)


def test_file_path(filename, nonexistent=None, not_ok_msg=None):
    """
    Test a file path, to see if it is exists and is a file
    :param filename: path to filesystem object to check
    :param nonexistent: message to display if nonexistent
    :param not_ok_msg: message to display if not a file
    :return: True if exists & is a file
    """
    return __test_path(filename, __ISFILE, nonexistent=nonexistent, not_ok_msg=not_ok_msg)


def test_dir_path(path, nonexistent=None, not_ok_msg=None):
    """
    Test a dir path, to see if it is exists and is a dir
    :param path: path to filesystem object to check
    :param nonexistent: message to display if nonexistent
    :param not_ok_msg: message to display if not a dir
    :return: True if exists & is a dir
    """
    return __test_path(path, __ISDIR, nonexistent=nonexistent, not_ok_msg=not_ok_msg)


def __test_path(path, chk_type, nonexistent=None, not_ok_msg=None):
    """
    Test a file path, to see if it is exists and is a file
    :param path: path to filesystem object to check
    :type chk_type: type to check for; __ISFILE or __ISDIR
    :param nonexistent: message to display if nonexistent
    :param not_ok_msg: message to display if not a file
    :return: True if exists & is the specified type
    """
    is_ok = False
    if path is not None:
        if not os.path.exists(path):
            if nonexistent is not None:
                print(nonexistent)
        else:
            if chk_type == __ISFILE:
                is_ok = os.path.isfile(path)
            elif chk_type == __ISDIR:
                is_ok = os.path.isdir(path)
            if not is_ok:
                if not_ok_msg is not None:
                    print(not_ok_msg)

    return is_ok
