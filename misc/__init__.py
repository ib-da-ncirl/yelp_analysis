# The MIT License (MIT)
# Copyright (c) 2020 Ian Buttimer

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

from .misc import less_dangerous_eval, ArgOptParam, default_or_val
from .misc_tf import pick_device, restrict_gpu_mem, get_optimiser, get_loss, predict, probability_to_class, \
    predict_img, get_conv2d, get_dense
from .arg_ctrl import ArgCtrl
from .image import resize_keep_aspect

__all__ = [
    'less_dangerous_eval',
    'ArgOptParam',
    'default_or_val',
    'pick_device',
    'restrict_gpu_mem',
    'get_optimiser',
    'get_loss',
    'predict',
    'probability_to_class',
    'predict_img',
    'get_conv2d',
    'get_dense',
    'ArgCtrl',
    'resize_keep_aspect',
]
