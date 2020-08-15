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
import string
from collections import namedtuple
from typing import Union


def less_dangerous_eval(equation):
    """
    Restrict expression that can be evaluated
    Thanks to https://stackoverflow.com/questions/19959333/convert-a-string-equation-to-an-integer-answer/19959928#19959928
    :param equation:
    :return:
    """
    # eval will run whatever it is given, even code, so it is dangerous to expose it to unsanitised input
    if not set(equation).intersection(string.ascii_letters + '{}[]_;\n'):
        return eval(equation)
    else:
        print("illegal character")
        return None


ArgOptParam = namedtuple('ArgOptParam', ['name', 'default'])


def default_or_val(param: ArgOptParam, dictionary: dict):
    """
    Get value for key from dictionary or return default
    :param param: option param
    :param dictionary: Dict to get value from
    :return:
    """
    return param.default if param.name not in dictionary else dictionary[param.name]


def decode_int_or_tuple(arg, tuple_size=2) -> Union[int, tuple, None]:
    result = None
    if ',' in arg:
        arg_split = arg.split(',')
        if len(arg_split) == tuple_size:
            result = tuple([int(x) for x in arg_split])
    else:
        result = int(arg)
    return result
