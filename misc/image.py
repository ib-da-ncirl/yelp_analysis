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

import os
from typing import Union

from PIL import Image, ImageOps


def resize_keep_aspect(path: str, desired_size: Union[int, tuple], out_folder: str):
    """
    Resize an image keeping the aspect ratio; resultant image will be 'desired_size' * 'desired_size' pixels
    :param path: Path to image
    :param desired_size: New size in pixels
    :param out_folder: folder to save new image
    """
    # Based on https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/

    im = Image.open(path)
    old_size = im.size  # old_size is in (width, height) format

    if isinstance(desired_size, tuple):
        ratio = [float(desired_size[i]/old_size[i]) for i in range(2)]
        ratio = min(ratio)
    else:
        ratio = float(desired_size) / max(old_size)
        desired_size = tuple([desired_size for i in range(2)])
    new_size = tuple([int(x * ratio) for x in old_size])
    coord = tuple([(desired_size[i] - new_size[i]) // 2 for i in range(2)])
    # use thumbnail() or resize() method to resize the input image

    # thumbnail is a in-place operation

    # im.thumbnail(new_size, Image.ANTIALIAS)

    im = im.resize(new_size, Image.ANTIALIAS)
    # create a new image and paste the resized on it

    new_im = Image.new("RGB", desired_size)
    new_im.paste(im, coord)

    new_im.save(os.path.join(out_folder, os.path.basename(path)))
    new_im.close()

