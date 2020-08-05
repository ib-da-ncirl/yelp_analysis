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

from .tf_image_eg import tf_image_eg
from .inception_v3 import inception_v3_eg, inception_v3_eg_v2
from .resnet50 import resnet50_eg
from .alexnet import alexnet
from .xception import xception_eg
from .model_args import ModelArgs

# Add the names of model functions to this list.
# Don't forget to import the function above as well
__all__ = [
    'ModelArgs',
    'tf_image_eg',
    'inception_v3_eg',
    'inception_v3_eg_v2',
    'resnet50_eg',
    'alexnet',
    'xception_eg',
]
