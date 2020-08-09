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

"""
Perform ETL on the Yelp Dataset CSV data to extract the subset of businesses/reviews etc. based on a parent category

Please see accompanying readme.md for detailed instructions.
"""

import os
import sys
import csv
import re
from math import ceil

import pandas as pd
import dask.dataframe as dd
import argparse
from timeit import default_timer as timer
from enum import Enum
from PIL import Image
from pathlib import Path
from collections import namedtuple

from misc import resize_keep_aspect

MIN_PYTHON = (3, 6)
if sys.version_info < MIN_PYTHON:
    sys.exit("Python %s.%s or later is required.\n" % MIN_PYTHON)


class Df(Enum):
    PANDAS = 1
    DASK = 2


def save_list(filepath, save_lst):
    """
    Save a list to file
    :param filepath: File to as as
    :param save_lst: List to save
    :return:
    """
    print(f"Saving '{filepath}'")
    with open(filepath, mode='w') as fh:
        fh.writelines(f"{entry}\n" for entry in save_lst)


def load_list(filepath):
    """
    Load a list from file
    :param filepath: File to load list from
    :return:
    """
    print(f"Loading '{filepath}'")
    with open(filepath, 'r') as fh:
        lst = [current_line.rstrip() for current_line in fh.readlines()]
    return lst


def check_parent(parent, alias, parent_aliases, exclude_lst):
    """
    Check if alias is equal to parent or a sub-alias
    :param parent:  parent alias to check for
    :param alias:   alias
    :param parent_aliases:  parent aliases
    :param exclude_lst:
    :return: True or False
    """
    candidate = (alias == parent or parent in parent_aliases)
    if candidate:
        candidate = (alias not in exclude_lst)
    return candidate


def verify_path(path, typ='file', create_dir=False):
    exists = os.path.exists(path)
    if typ == 'folder':
        if not exists and create_dir:
            Path(path).mkdir(parents=True, exist_ok=True)
            exists = True
        valid = os.path.isdir(path)
    elif typ == 'file':
        valid = os.path.isfile(path)
    else:
        raise ValueError(f"Unrecognised 'typ' argument: {typ}")
    if not exists:
        error(f"'{path}' does not exist")
    if not valid:
        error(f"'{path}' is not a {typ}")
    return exists and valid


def get_categories(category_path, parent_category, exclude_str, save_cats_path, verbose):
    """
    Process the categories file to get a list of all sub-categories of the specified parent
    :param category_path: Path to categories json file
    :param parent_category: Parent category
    :param exclude_str: Comma separated string of categories to exclude
    :param save_cats_path: File to save category list to
    :param verbose: Verbose mode flag
    :return: List of categories
    """

    print(f"Retrieving sub-categories of '{parent_category}' from '{category_path}'")

    # read business categories
    # e.g. {
    #         "alias": "burgers",
    #         "title": "Burgers",
    #         "parent_aliases": [
    #             "restaurants"
    #         ],
    #         "country_whitelist": [],
    #         "country_blacklist": []
    # },
    cat_df = pd.read_json(category_path)

    excludes = []
    if exclude_str is not None:
        if len(exclude_str) > 0:
            excludes = exclude_str.split(',')

    # set 'req' column; true for aliases with parent_category as parent
    cat_df['req'] = cat_df['categories'].apply(
        lambda lst: check_parent(parent_category, lst['alias'], lst['parent_aliases'], excludes))
    # get just the alias
    req_cats_df = cat_df[cat_df['req']]['categories'].apply(lambda lst: lst['title'])
    req_cats_lst = req_cats_df.to_list()

    print(f"{len(req_cats_lst)} sub-categories identified")
    if len(req_cats_lst) == 0:
        print(f"Please verify '{parent_category}' is a valid alias")
    elif verbose:
        print(f"{req_cats_lst}")

    if save_cats_path is not None:
        save_list(save_cats_path, req_cats_lst)

    return req_cats_lst


def check_categories(categories_str, category_list):
    """
    Check if at least one entry in category_list is in the category string
    :param categories_str: Comma-separated list of categories
    :param category_list: List of categories to match
    :return:
    """
    req = False
    if isinstance(categories_str, str):
        categories = categories_str.split(',')
        for category in categories:
            req = category.strip() in category_list
            if req:
                break
    return req


def duration(start, verbose):
    if verbose:
        print(f"Duration: {timer() - start:.1f}s")


def save_csv(df, out_path, index=False):
    """
    Save a Dataframe to csv
    :param df: Dataframe to save
    :param out_path: Path to save to
    :param index: Write row names (index)
    :return:
    """
    if out_path is not None:
        print(f"Saving {len(df)} rows to '{out_path}'")
        df.to_csv(out_path, index=index)


def binary_search(arr, low, high, x):
    """
    Find index of x in arr if present, else -1
    Thanks to https://www.geeksforgeeks.org/python-program-for-binary-search/
    :param arr: Ascending order sorted array to search
    :param low: Start index (inclusive)
    :param high: End index (inclusive)
    :param x: Element to find
    :return: index of x in arr if present, else -1
    """
    # Check base case
    if high >= low:
        mid = (high + low) // 2
        # If element is present at the middle itself
        if arr[mid] == x:
            return mid
        # If element is smaller than mid, then it can only be present in left subarray
        elif arr[mid] > x:
            return binary_search(arr, low, mid - 1, x)
        # Else the element can only be present in right subarray
        else:
            return binary_search(arr, mid + 1, high, x)
    else:
        # Element is not present in the array
        return -1


def load_csv(entity_name, csv_path, dtype=None, converters=None, out_path=None,
             filter_id_column=None, filter_id_lst=None,
             filter_lambda_column=None, filter_lambda=None,
             entity_id_column=None,
             arg_dict=None, csv_id=None):
    """
    Get the entities for the specified list of business ids
    :param entity_name: Name of entity being processed
    :param csv_path: Path to entity csv file
    :param dtype: dtypes for read_csv
    :param converters: converters for read_csv
    :param out_path: Path to save filtered entities to
    :param filter_id_column: Name of id column to filter by
    :param filter_id_lst:  List of ids to filter by
    :param filter_lambda_column:  Name of column to apply lambda filter to
    :param filter_lambda:   lambda filter function
    :param entity_id_column:    Name of id column of the entity
    :param arg_dict: keyword arguments
                     - 'verbose': verbose mode flag; True/False
                     - 'dataframe': dataframe type; Df.PANDAS/Df.DASK, default Df.PANDAS
                     - 'col': name of column to filter on
                     - 'regex': regex to filter column 'col'
                     - 'drop_regex': regex for columns to drop
                     - 'nrows': number of rows to read, (Df.PANDAS only)
                     - 'show_duration': display duration flag in verbose mode; True/False
    :param csv_id: cvs_id of arg_dict['regex'] to use for column matching
    :return:
    """
    start = timer()

    if arg_dict is None:
        arg_dict = {}
    verbose = False if 'verbose' not in arg_dict else arg_dict['verbose']
    df = Df.PANDAS if 'dataframe' not in arg_dict else arg_dict['dataframe']
    drop_regex = None if 'drop_regex' not in arg_dict else arg_dict['drop_regex']
    limit_id = None if 'limit_id' not in arg_dict else arg_dict['limit_id']
    # exclude id list flag; default False. False: accept all ids in list, True: accept all ids not in list
    ex_id = False if 'ex_id' not in arg_dict else arg_dict['ex_id']
    show_duration = True if 'show_duration' not in arg_dict else arg_dict['show_duration']
    parse_engine = 'python' if 'parse_engine' not in arg_dict else arg_dict['parse_engine']

    col = None
    regex = None
    if 'regex' in arg_dict and csv_id is not None:
        if csv_id in arg_dict['regex']:
            col = arg_dict['regex'][csv_id][0]
            regex = arg_dict['regex'][csv_id][1]

    do_filter_col = (col is not None and regex is not None)  # do col/regex filter
    do_filter_lambda = (filter_lambda_column is not None and filter_lambda is not None)  # do lambda filter?

    # hash ids for faster comparison
    hash_id_list = None
    do_filter_by_id = (filter_id_lst is not None and filter_id_column is not None)  # do filter id?
    if do_filter_by_id:
        hash_id_list = []
        for bid in filter_id_lst:
            hash_id_list.append(hash(bid))
        hash_id_list = sorted(hash_id_list)

    if limit_id is not None:
        max_count = limit_id
    else:
        max_count = sys.maxsize

    print(f"Reading '{csv_path}'")
    if verbose:
        print(f"Using {df}")

    parse_args = {
        'dtype': dtype,
        'converters': converters,
        'encoding': 'utf-8',
        'engine': parse_engine,
        'quoting': csv.QUOTE_MINIMAL
    }
    try:
        if df == Df.PANDAS:
            if 'nrows' in arg_dict:
                parse_args['nrows'] = arg_dict['nrows']
            rev_df = pd.read_csv(csv_path, **parse_args)
        elif df == Df.DASK:
            if 'nrows' in arg_dict:
                warning("-nr/--nrows not supported for Dask dataframe")
            rev_df = dd.read_csv(csv_path, **parse_args)
        else:
            raise NotImplemented(f'Unrecognised dataframe type: {df}')
    except pd.errors.ParserError:
        error(f"The current max csv field size of {hex(csv.field_size_limit() // 0x1000)[2:]}kB is too small. "
              f"Use the '-cs/--csv_size' option to increase the max csv field size")
        rev_df = None

    print(f"{len(rev_df)} {entity_name} loaded")

    if do_filter_col or do_filter_lambda or do_filter_by_id or (limit_id is not None):
        # filter by column value
        cmts = []
        if do_filter_col:
            if col not in rev_df.columns:
                error(f"Column '{col}' not in dataframe")

            cmts.append(f"'{col}' column")

            def col_val_fxn(mcol):
                return True if re.search(regex, mcol) else False
        else:
            def col_val_fxn(mcol):
                return True

        if filter_lambda is not None:
            cmts.append(f"'{filter_lambda_column}' column")
        if do_filter_by_id is not False:
            cmts.append(f"'{filter_id_column}' column, {len(hash_id_list)} possible values")

        count = 0
        total = len(rev_df)
        if df != Df.DASK:
            def inc_progress():
                nonlocal count
                count = progress("Row", total, count)
        else:
            def inc_progress():
                nonlocal count
                count += 1

        def filter_by_id_and_col(filter_props):
            if do_filter_lambda:
                # filter on lambda function
                ok = filter_lambda(filter_props[filter_lambda_column])
            else:
                ok = True
            if ok:
                # filter on column value
                ok = col_val_fxn(None if col is None else filter_props[col])
            if ok:
                if do_filter_by_id:
                    # filter on ids
                    ok = (binary_search(hash_id_list, 0, len(hash_id_list) - 1,
                                        hash(filter_props[filter_id_column])) >= 0)
                if ex_id:
                    ok = not ok  # exclude id
            nonlocal max_count
            if ok and max_count > 0:
                max_count -= 1
            else:
                ok = False
            inc_progress()
            return ok

        cmt = f"Filtering for required {entity_name}"
        for cnd in cmts:
            cmt += f"\n- matching {cnd}"
        if max_count < sys.maxsize:
            cmt += f"\n- max count {max_count}"
        print(cmt)

        # filter by ids
        filter_col_list = [filter_id_column, col] if col is not None else [filter_id_column]
        if filter_lambda_column is not None and filter_lambda is not None:
            filter_col_list.append(filter_lambda_column)
        if df == Df.PANDAS:
            rev_df['req'] = rev_df[filter_col_list].apply(filter_by_id_and_col, axis=1)
        elif df == Df.DASK:
            rev_df['req'] = rev_df[filter_col_list].apply(filter_by_id_and_col, axis=1, meta=('req', 'bool'))

        if df != Df.DASK:
            progress("Row", total, total)

    # drop columns matching the drop column regex
    if drop_regex is not None:
        cols_to_drop = set()
        for dcol in rev_df.columns.values:
            if re.search(drop_regex, dcol):
                cols_to_drop.add(dcol)

        if len(cols_to_drop) > 0:
            print(f'Dropping {len(cols_to_drop)} unnecessary columns')
            print(f'{cols_to_drop}')
            rev_df = rev_df.drop(list(cols_to_drop), axis=1)

    # get just the required rows
    if 'req' in rev_df.columns.values:
        req_rev_df = rev_df[rev_df['req']]
        # drop work column
        req_rev_df = req_rev_df.drop(['req'], axis=1)
    else:
        req_rev_df = rev_df

    if entity_id_column is not None:
        entity_id_lst = req_rev_df[entity_id_column].to_list()
    else:
        entity_id_lst = None

    print(f"{len(req_rev_df)} {entity_name} identified")

    save_csv(req_rev_df, out_path, index=False)

    if show_duration:
        duration(start, verbose)

    return req_rev_df, entity_id_lst, start


def get_business(csv_path, category_list=None, out_path=None, save_ids_path=None, id_list=None, arg_dict=None,
                 csv_id='biz'):
    """
    Process the businesses csv file to get a list of businesses with the specified categories
    :param csv_path: Path to business csv file
    :param category_list: List of categories to match
    :param out_path: File to save filtered businesses to
    :param save_ids_path: File to save business id list to
    :param id_list: List of business ids
    :param arg_dict: See load_csv::arg_dict
    :param csv_id: cvs_id of arg_dict['regex'] to use for column matching
    :return:  Business id list
    """

    arg_dict['show_duration'] = False
    verbose = False if 'verbose' not in arg_dict else arg_dict['verbose']

    # read business details
    # - DtypeWarning: Columns (36,37,41,43,46,56,67,72,76,80,99) have mixed types
    #   most of the columns have bool mixed with missing values so use a converter to preserve the missing value
    #   for the moment
    def bool_or_none(x):
        return bool(x) if x else None

    if category_list is not None:
        def filter_lambda_fxn(lst):
            return check_categories(lst, category_list)

        filter_lambda = filter_lambda_fxn
    else:
        filter_lambda = None

    biz_df, req_biz_id_lst, start = load_csv("businesses", csv_path, dtype={
        "attributes.AgesAllowed": str,  # 36
        "attributes.DietaryRestrictions": str,  # 41
    }, converters={
        "attributes.DietaryRestrictions.gluten-free": bool_or_none,  # 37
        "attributes.DietaryRestrictions.vegan": bool_or_none,  # 43
        "attributes.DietaryRestrictions.dairy-free": bool_or_none,  # 46
        "attributes.DietaryRestrictions.halal": bool_or_none,  # 56
        "attributes.DietaryRestrictions.soy-free": bool_or_none,  # 67
        "attributes.DietaryRestrictions.vegetarian": bool_or_none,  # 72
        "attributes.RestaurantsCounterService": bool_or_none,  # 76
        "attributes.DietaryRestrictions.kosher": bool_or_none,  # 80
        "attributes.Open24Hours": bool_or_none,  # 99
    }, filter_id_column='business_id', filter_id_lst=id_list,
                                             filter_lambda_column='categories',
                                             filter_lambda=filter_lambda,
                                             entity_id_column='business_id', arg_dict=arg_dict, csv_id=csv_id)

    # find root columns which are not required as values have been expanded into their own columns
    # e.g. 'hours' is not required as info is in 'hours.Friday' etc.
    col_to_drop = set()
    for col in biz_df.columns.values:
        if "." in col:
            dot_splits = col.split(".")
            col_name = dot_splits[0]
            col_to_drop.add(col_name)
            for col_idx in range(1, len(dot_splits) - 1):
                col_name = col_name + '.' + dot_splits[col_idx]
                col_to_drop.add(col_name)

    if len(col_to_drop) > 0:
        print(f'Dropping {len(col_to_drop)} unnecessary columns')
        print(f'{col_to_drop}')
        biz_df = biz_df.drop(list(col_to_drop), axis=1)

    biz_df = biz_df.fillna('')

    duration(start, verbose)

    save_csv(biz_df, out_path, index=False)

    if save_ids_path is not None:
        save_list(save_ids_path, req_biz_id_lst)

    return biz_df, req_biz_id_lst


def get_reviews(csv_path, id_lst, out_path, prefilter_path=None, arg_dict=None):
    """
    Get the reviews for the specified list of business ids
    :param csv_path: Path to review csv file
    :param id_lst:  List of business ids
    :param out_path: Path to save filtered reviews to
    :param prefilter_path: Path to save pre-filtered reviews to
    :param arg_dict: keyword arguments
    :return:
    """
    load_path = csv_path
    filter_id_lst = id_lst
    if prefilter_path is not None:
        load_path = prefilter_path  # give pre-filtered input to load_csv()
        filter_id_lst = None  # no need to filter ids in load_csv()
        biz_iz_idx = -1

        # hash ids for faster comparison
        hash_id_list = None
        if id_lst is not None:
            hash_id_list = []
            for bid in id_lst:
                hash_id_list.append(hash(bid))

        print(f"Pre-filter {csv_path} to {prefilter_path}")

        count = 0
        total = 0
        with open(csv_path, "r") as fhin:
            with open(prefilter_path, "w") as fhout:
                for line in fhin:
                    columns = line.strip(' \n').split(",")
                    if count == 0:
                        if "business_id" in columns:
                            biz_iz_idx = columns.index("business_id")
                        if biz_iz_idx < 0:
                            error("'business_id' index not found")
                        ok = True
                    else:
                        ok = hash(columns[biz_iz_idx]) in hash_id_list

                    count = progress("Row", total, count)

                    if ok:
                        fhout.write(line)

    return load_csv('reviews', load_path, dtype={
        "cool": object,
        "funny": object,
        "useful": object,
        "stars": object,
    }, out_path=out_path, filter_id_column='business_id', filter_id_lst=filter_id_lst,
                    arg_dict=arg_dict, csv_id='review')


def get_tips(csv_path, id_lst, out_path, arg_dict=None):
    """
    Get the tips for the specified list of business ids
    :param csv_path: Path to tip csv file
    :param id_lst:  List of business ids
    :param out_path: Path to save filtered reviews to
    :param arg_dict: keyword arguments
    :return:
    """
    return load_csv('tips', csv_path, dtype={
        "compliment_count": object,
    }, out_path=out_path, filter_id_column='business_id', filter_id_lst=id_lst, arg_dict=arg_dict, csv_id='tips')


def get_checkin(csv_path, id_lst, out_path, arg_dict=None):
    """
    Get the checkin for the specified list of business ids
    :param csv_path: Path to checkin csv file
    :param id_lst:  List of business ids
    :param out_path: Path to save filtered reviews to
    :param arg_dict: keyword arguments
    :return:
    """
    return load_csv('checkins', csv_path, out_path=out_path, filter_id_column='business_id', filter_id_lst=id_lst,
                    arg_dict=arg_dict)


def get_photos(csv_path, id_lst, out_path, arg_dict=None):
    """
    Get the photos for the specified list of business ids
    :param csv_path: Path to checkin csv file
    :param id_lst:  List of business ids
    :param out_path: Path to save filtered photos to
    :param arg_dict: keyword arguments
    :return:
    """
    return load_csv('photos', csv_path, out_path=out_path, filter_id_column='business_id', filter_id_lst=id_lst,
                    arg_dict=arg_dict, csv_id='pin')


def progress(cmt, total, current, step=100):
    current += 1
    if current % step == 0 or total == current:
        percent = "" if total == 0 else f"({current * 100 / total:.1f}%)"
        print(f"{cmt}: {current} {percent}", flush=True, end='\r' if total > current or total == 0 else '\n')
    return current


def img_process(photo, extensions, photo_folder, count, total,
                get_attrib=False, resize=False, size=None, resize_folder=None):
    """
    Process photos
    :param photo: Image name
    :param extensions: List of possible extensions
    :param photo_folder: Folder containing image files
    :param count: Current processed count
    :param total: Total number to process
    :param get_attrib: Get attributes flag
    :param resize: Resize image flag
    :param size: Size to resize to
    :param resize_folder: Folder to save resized images
    :return:
    """
    attrib = None
    for ext in extensions:
        filename = f"{photo}{ext}"
        filepath = os.path.join(photo_folder, filename)
        if os.path.isfile(filepath):
            if get_attrib:
                im = Image.open(filepath)
                attrib = f"{filename},{im.format},{im.size[0]},{im.size[1]},{im.mode}"
                im.close()
            else:
                attrib = None

            if resize:
                resize_keep_aspect(filepath, size, resize_folder)

            count = progress("Photo", total, count)
            break
    if get_attrib and attrib is None:
        raise ValueError(f"Unmatched photo id {photo}")
    return count, attrib


def generate_photo_set(biz_path, photo_csv_path, id_lst, photo_folder, out_path, save_ids_path=None, arg_dict=None):
    """
    Generate a csv for the photo dataset
    :param biz_path: Path to business csv file
    :param photo_csv_path: Path to photo csv file
    :param id_lst:  List of business ids
    :param photo_folder: Path to folder containing photos
    :param out_path: Path to save dataset to
    :param save_ids_path: File to save business id list to
    :param arg_dict: keyword arguments
    :return:
    """
    biz_df, _ = get_business(biz_path, id_list=id_lst, arg_dict=arg_dict, csv_id='biz_photo')
    photo_df, _, _ = get_photos(photo_csv_path, id_lst, None, arg_dict=arg_dict)

    # categorical representation; e.g. '1_0' represents 1.0 stars
    biz_df['stars_cat'] = biz_df['stars'].apply(lambda x: str(x).replace(".", "_"))
    # ordinal representation;
    # e.g. [0,0,0,0,0,0,0,0,0] represents 1.0 star, [1,0,0,0,0,0,0,0,0] represents 1.5 stars etc.
    star_min = biz_df['stars'].min()

    max_len = ceil(((biz_df['stars'].max() - star_min) * 2)) + 1    # stars range * num of 0.5 + 1

    def fill_array(x):
        one_cnt = ceil((x - star_min) * 2)
        return ([1] * one_cnt) + ([0] * (max_len - one_cnt))
    biz_df['stars_ord'] = biz_df['stars'].apply(fill_array)

    # numerical representation; e.g. 0.5 = 1, 1.0 = 2 etc.
    biz_df['stars_num'] = biz_df['stars'].apply(lambda x: ceil(x * 2))

    # many-to-one join, i.e many photos to one business id
    biz_photo_df = pd.merge(photo_df, biz_df, on='business_id', how='outer')

    # drop rows with no photo id, no photo no prediction
    total = len(biz_photo_df)
    biz_photo_df = biz_photo_df[~biz_photo_df['photo_id'].isna()]

    if total > len(biz_photo_df):
        print(f"Dropped {total - len(biz_photo_df)} businesses with no photos")

    if 'random_select' in arg_dict:
        sample_ctrl = {
            'n': int(arg_dict['random_select']) if arg_dict['random_select'] >= 1.0 else None,
            'frac': arg_dict['random_select'] if arg_dict['random_select'] < 1.0 else None
        }
        pre_sample_len = len(biz_photo_df)
        if arg_dict['select_on'].lower() == 'all':
            # do sample on all photos available
            biz_photo_df = biz_photo_df.sample(**sample_ctrl, random_state=1)

            print(f"Sampled {len(biz_photo_df)} from {pre_sample_len} possible photos")
        else:
            # do sample on unique values of specified column
            # make sorted list of hashed unique samples
            unique_biz_ids = pd.Series(biz_photo_df[arg_dict['select_on']].unique())
            vals_to_match = sorted(unique_biz_ids.
                                   sample(**sample_ctrl, random_state=1).apply(hash).to_list())

            def is_req(row):
                return binary_search(vals_to_match, 0, len(vals_to_match) - 1, hash(row)) >= 0

            biz_photo_df = biz_photo_df[biz_photo_df[arg_dict['select_on']].apply(is_req)]

            print(f"Sampled {len(vals_to_match)} from {len(unique_biz_ids)} "
                  f"possible {arg_dict['select_on']}, giving {len(biz_photo_df)} photos")

    # process photos
    extensions = Image.registered_extensions().keys()
    count = 0
    total = len(biz_photo_df)

    # process photos details
    print(f"Processing photo details")
    resize = ('photo_folder_resize' in arg_dict and 'photo_size' in arg_dict)
    if resize:
        print(f"  Including resizing photos to {arg_dict['photo_size']}x{arg_dict['photo_size']}px in "
              f"{arg_dict['photo_folder_resize']}")
        Path(arg_dict['photo_folder_resize']).mkdir(parents=True, exist_ok=True)

        def img_attrib(photo):
            nonlocal count
            count, attrib = img_process(photo, extensions, photo_folder, count, total, get_attrib=True, resize=True,
                                        size=arg_dict['photo_size'], resize_folder=arg_dict['photo_folder_resize'])
            return attrib
    else:
        def img_attrib(photo):
            nonlocal count
            count, attrib = img_process(photo, extensions, photo_folder, count, total, get_attrib=True)
            return attrib

    biz_photo_df['photo_attrib'] = biz_photo_df['photo_id'].apply(img_attrib)
    biz_photo_df[['photo_file', 'format', 'width', 'height', 'mode']] = \
        biz_photo_df.apply(lambda row: pd.Series(row['photo_attrib'].split(',')), axis=1, result_type='expand')

    biz_photo_df['width'] = biz_photo_df['width'].astype('int32')
    biz_photo_df['height'] = biz_photo_df['height'].astype('int32')

    progress("Photo", len(biz_photo_df), len(biz_photo_df))

    # just keep required columns
    photo_set_df = biz_photo_df[['business_id', 'stars', 'stars_cat', 'stars_ord', 'stars_num', 'photo_id',
                                 'photo_file', 'format', 'width', 'height', 'mode']]

    def wh_anal(column):
        lwr = column.lower()
        print(f"{column}: min - {photo_set_df[lwr].min()}px, max - {photo_set_df[lwr].max()}px")
        print(f"{column}: value counts - {photo_set_df[lwr].value_counts()}")

    unique_vals = photo_set_df['format'].unique()
    print(f"Format: {len(unique_vals)} format{'' if len(unique_vals) == 1 else 's'} - {unique_vals}")
    wh_anal("Width")
    wh_anal("Height")
    unique_vals = photo_set_df['mode'].unique()
    print(f"Mode: {len(unique_vals)} mode{'' if len(unique_vals) == 1 else 's'} - {unique_vals}")
    unique_vals = photo_set_df['stars'].unique()
    unique_cat_vals = photo_set_df['stars_cat'].unique()
    unique_num_vals = photo_set_df['stars_num'].unique()
    unique_ord_vals = set()
    photo_set_df['stars_ord'].apply(lambda x: unique_ord_vals.add(tuple(x)))
    print(f"Stars: {len(unique_vals)} class{'' if len(unique_vals) == 1 else 'es'} - {unique_vals}\n"
          f"\t{unique_cat_vals}\n\t{unique_num_vals}\n\t{unique_ord_vals}")

    if save_ids_path is not None:
        save_list(save_ids_path, biz_photo_df['business_id'].unique().tolist())

    save_csv(photo_set_df, out_path, index=False)


def resize_photo_set(dataset_csv_path, photo_folder, arg_dict=None):
    """
    Generate a csv for the photo dataset
    :param dataset_csv_path: Path to dataset csv file
    :param photo_folder: Path to folder containing photos
    :param arg_dict: keyword arguments
    :return:
    """

    photo_set_df, _, start = load_csv('photos', dataset_csv_path, arg_dict=arg_dict)

    print(f"Resizing photos to {arg_dict['photo_size']}x{arg_dict['photo_size']}px in "
          f"{arg_dict['photo_folder_resize']}")

    Path(arg_dict['photo_folder_resize']).mkdir(parents=True, exist_ok=True)

    # process photos
    extensions = Image.registered_extensions().keys()
    count = 0
    total = len(photo_set_df)

    def img_proc(photo):
        nonlocal count
        count, _ = img_process(photo, extensions, photo_folder, count, total, resize=True,
                               size=arg_dict['photo_size'], resize_folder=arg_dict['photo_folder_resize'])

    photo_set_df['photo_id'].apply(img_proc)

    duration(start, verbose=False if 'verbose' not in arg_dict else arg_dict['verbose'])


def file_arg_help(descrip, target='file', action=None):
    if action is None:
        tail = ';'
    else:
        tail = f' {action};'
    return f"Path to {descrip} {target}{tail} absolute or relative to 'root directory' if argument supplied"


def warning(msg):
    print(f"Warning: {msg}")


def error(msg):
    sys.exit(f"Error: {msg}")


def ignore_arg_warning(args_namespace, arg_lst):
    for arg in arg_lst:
        if arg in args_namespace:
            warning(f"Ignoring '{arg}' argument")


def arg_error(arg_parser, msg):
    arg_parser.print_usage()
    sys.exit(msg)


def required_arg_error(arg_parser, req_arg):
    arg_error(arg_parser, f"{os.path.split(sys.argv[0])[1]}: error one of the arguments {req_arg} is required")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Perform ETL on the Yelp Dataset CSV data to extract the subset of businesses/reviews etc. '
                    'based on a parent category',
    )
    parser.add_argument(
        '-d', '--dir',
        type=str,
        help='Root directory',
        default='',
    )
    # business csv file or business ids file
    biz_group = parser.add_mutually_exclusive_group(required=False)
    biz_group.add_argument('-b', '--biz', type=str, help=file_arg_help("business csv"), default='')
    biz_group.add_argument('-bi', '--biz_ids', type=str, help=file_arg_help("business ids"), default='')
    # review, tips, checkin & photo input csv files
    for arg_tuple in [('-r', '--review', 'review'),
                      ('-t', '--tips', 'tips'),
                      ('-ci', '--chkin', 'checkin'),
                      ('-pi', '--pin', 'photo'),
                      ('-psi', '--photo_set_in', 'photo dataset')]:
        parser.add_argument(
            arg_tuple[0], arg_tuple[1], type=str, help=file_arg_help(f"{arg_tuple[2]} csv"), default='', required=False
        )
    # categories json file or categories file
    cat_group = parser.add_mutually_exclusive_group(required=False)
    cat_group.add_argument('-c', '--cat', type=str, help=file_arg_help("categories json"), default='')
    cat_group.add_argument('-cl', '--cat_list', type=str, help=file_arg_help("category list"), default='')
    # category selection
    parser.add_argument('-p', '--parent', type=str, help='Parent category', default='', required=False)
    parser.add_argument(
        '-e', '--exclude', type=str, help='Exclude categories; a comma separated list of categories to exclude',
        default='', required=False)
    # output files
    for arg_tuple in [('-ob', '--out_biz', 'business'),
                      ('-opr', '--out_prefilter_review', 'review pre-filter'),
                      ('-or', '--out_review', 'review'),
                      ('-ot', '--out_tips', 'tips'),
                      ('-oci', '--out_chkin', 'checkin'),
                      ('-op', '--out_photo', 'photo'),
                      ('-ops', '--out_photo_set', 'photo set')]:
        parser.add_argument(
            arg_tuple[0], arg_tuple[1], type=str, help=file_arg_help(f"{arg_tuple[2]} csv", action="to create"),
            default=argparse.SUPPRESS, required=False
        )
    parser.add_argument('-bp', '--biz_photo', type=str, help=file_arg_help("business csv for photo dataset"),
                        default='')  # just so as not to conflict with mutually exclusive --biz/--biz_ids arguments
    for arg_tuple in [('-oc', '--out_cat', 'category list'),
                      ('-obi', '--out_biz_id', 'business ids')]:
        parser.add_argument(
            arg_tuple[0], arg_tuple[1], type=str, help=file_arg_help(f"{arg_tuple[2]}", action="to create"),
            default=argparse.SUPPRESS, required=False
        )
    # input folders
    parser.add_argument('-pf', '--photo_folder', type=str, help=file_arg_help("photo", target='folder'),
                        default='')
    # output folders
    parser.add_argument('-pfr', '--photo_folder_resize', type=str,
                        help=file_arg_help("resized photos", target='folder'),
                        default='')
    # miscellaneous
    parser.add_argument('-dx', '--drop_regex', help='Regex for business csv columns to drop', type=str, default=None)
    parser.add_argument('-mx', '--match_regex', help="Regex for csv columns to match; 'csv_id:column_name=regex'. "
                                                     "Valid 'csv_id' are; 'biz'=business csv file, "
                                                     "'pin'=photo csv file, 'tip'=tip csv file and "
                                                     "'review'=review csv file",
                        type=str, default=argparse.SUPPRESS, nargs='+')
    parser.add_argument('-df', '--dataframe', help="Dataframe to use; 'pandas' or 'dask'",
                        choices=['pandas', 'dask'], required=False)
    parser.add_argument('-pe', '--parse_engine', help="Parser engine to use; 'c' or 'python'}",
                        choices=['c', 'python'], required=False)
    parser.add_argument('-nr', '--nrows', help="Number of rows to read, (Note: ignored with '-df=dask' option)",
                        type=int, default=argparse.SUPPRESS)
    parser.add_argument('-li', '--limit_id', help="Limit number of business ids to read",
                        type=int, default=argparse.SUPPRESS)
    parser.add_argument('-rs', '--random_select', help="Make random selection; 'value' < 1.0 = percent of total "
                                                       "available, or 'value' > 1 = number to select",
                        type=float, default=argparse.SUPPRESS)
    parser.add_argument('-so', '--select_on', help="Column to make selection on or 'all' to select from total "
                                                   "available; e.g. 'business_id'",
                        type=str, default=argparse.SUPPRESS)
    parser.add_argument('-cs', '--csv_size',
                        help=f"max csv field size in kB; default {hex(csv.field_size_limit() // 0x1000)[2:]}kB",
                        type=int, default=argparse.SUPPRESS)
    parser.add_argument('-ps', '--photo_size',
                        help=f"required photo size in pixels",
                        type=int, default=argparse.SUPPRESS)
    parser.add_argument('-v', '--verbose', help='Verbose mode', action='store_true')

    args = parser.parse_args()

    if len(sys.argv) == 1:
        # display help message when no args are passed.
        parser.print_help()
        sys.exit(1)

    if args.verbose:
        print(f"Arguments: {args}")

    if 'csv_size' in args:
        csv.field_size_limit(int(str(args.csv_size), 16) * 0x1000)  # convert kB to bytes

    paths = {'biz': None, 'biz_ids': None, 'biz_photo': None, 'photo_folder': None, 'photo_folder_resize': None,
             'cat': None, 'cat_list': None,
             'review': None, 'tips': None, 'chkin': None, 'pin': None, 'photo_set_in': None,
             'out_biz': None, 'out_review': None, 'out_prefilter_review': None,
             'out_tips': None, 'out_chkin': None, 'out_photo': None,
             'out_photo_set': None,
             'out_cat': None, 'out_biz_id': None}
    IpArgDetail = namedtuple('IpArgDetail', ['name', 'value', 'typ', 'create_dir'])
    ip_arg_tuples = [IpArgDetail('biz', args.biz, 'file', False), IpArgDetail('biz_ids', args.biz_ids, 'file', False),
                     IpArgDetail('biz_photo', args.biz_photo, 'file', False),
                     IpArgDetail('photo_folder', args.photo_folder, 'folder', False),
                     IpArgDetail('photo_folder_resize', args.photo_folder_resize, 'folder', True),
                     IpArgDetail('cat', args.cat, 'file', False), IpArgDetail('cat_list', args.cat_list, 'file', False)]
    for arg_tuple in ip_arg_tuples:
        if arg_tuple.name in args:
            paths[arg_tuple.name] = arg_tuple.value
    for arg_tuple in [('review', args.review),
                      ('tips', args.tips),
                      ('chkin', args.chkin),
                      ('pin', args.pin),
                      ('photo_set_in', args.photo_set_in)]:
        if arg_tuple[0] in args and len(arg_tuple[1]):
            paths[arg_tuple[0]] = arg_tuple[1]
    for arg_tuple in [('out_biz', None if 'out_biz' not in args else args.out_biz),
                      ('out_prefilter_review',
                       None if 'out_prefilter_review' not in args else args.out_prefilter_review),
                      ('out_review', None if 'out_review' not in args else args.out_review),
                      ('out_tips', None if 'out_tips' not in args else args.out_tips),
                      ('out_chkin', None if 'out_chkin' not in args else args.out_chkin),
                      ('out_photo', None if 'out_photo' not in args else args.out_photo),
                      ('out_photo_set', None if 'out_photo_set' not in args else args.out_photo_set),
                      ('out_cat', None if 'out_cat' not in args else args.out_cat),
                      ('out_biz_id', None if 'out_biz_id' not in args else args.out_biz_id)]:
        if arg_tuple[0] in args:
            paths[arg_tuple[0]] = arg_tuple[1]
    if len(args.dir) > 0:
        for key, val in paths.items():
            if val is not None and not os.path.isabs(val):
                paths[key] = os.path.join(args.dir, val)

    for arg_tuple in ip_arg_tuples:
        if len(arg_tuple.value) > 0:
            verify_path(paths[arg_tuple.name], typ=arg_tuple.typ, create_dir=arg_tuple.create_dir)

    kwarg_dict = {'verbose': args.verbose, 'drop_regex': args.drop_regex}
    if args.dataframe is not None:
        kwarg_dict['dataframe'] = Df.PANDAS if args.dataframe == 'pandas' else Df.DASK
    if args.parse_engine is not None:
        kwarg_dict['parse_engine'] = 'c' if args.parse_engine == 'c' else 'python'
    for arg_tuple in [('limit_id', None if 'limit_id' not in args else args.limit_id),
                      ('nrows', None if 'nrows' not in args else args.nrows)]:
        if arg_tuple[0] in args:
            kwarg_dict[arg_tuple[0]] = arg_tuple[1]
    if 'match_regex' in args:
        for regex_cmd in args.match_regex:
            split_file_regex = regex_cmd.split(':')
            if len(split_file_regex) != 2:
                arg_error(parser, f"Invalid format for -mx/--match_regex, expected 'csv_id:column_name=regex'")

            splits_col_regex = split_file_regex[1].split('=')
            if len(splits_col_regex) != 2:
                arg_error(parser, f"Invalid format for -mx/--match_regex, expected 'csv_id:column_name=regex'")

            if 'regex' not in kwarg_dict:
                kwarg_dict['regex'] = {}
            #                   csv_id                [column_name, regex]
            kwarg_dict['regex'][split_file_regex[0]] = splits_col_regex
    if 'photo_folder_resize' in args or 'photo_size' in args:
        if ('photo_size' in args and len(args.photo_folder_resize) == 0) or \
           ('photo_size' not in args and len(args.photo_folder_resize)):
            arg_error(parser, f"Options -pfr/--photo_folder_resize and -ps/--photo_size are both required")
        if 'photo_size' in args and len(args.photo_folder_resize):
            kwarg_dict['photo_folder_resize'] = paths['photo_folder_resize']
            kwarg_dict['photo_size'] = args.photo_size
    if 'random_select' in args or 'select_on' in args:
        if (('random_select' in args and 'select_on' not in args) or
                ('random_select' not in args and 'select_on' in args)):
            arg_error(parser, f"Options -rs/--random_select and -so/--select_on are both required")
        kwarg_dict['random_select'] = args.random_select
        kwarg_dict['select_on'] = args.select_on

    # load categories
    if len(args.cat_list) > 0:
        ignore_arg_warning(args, ['out_cat', 'parent', 'exclude'])
        categories_lst = load_list(paths['cat_list'])
    elif len(args.cat) > 0:
        categories_lst = get_categories(paths['cat'], args.parent, args.exclude, paths['out_cat'], args.verbose)
    else:
        categories_lst = None

    # load business ids
    if len(args.biz_ids) > 0:
        ignore_arg_warning(args, ['out_biz', 'out_biz_id'])
        biz_id_lst = load_list(paths['biz_ids'])
    elif len(args.biz) > 0:
        if categories_lst is None:
            required_arg_error(parser, ['-c/--cat', '-cl/--cat_list'])
        _, biz_id_lst = get_business(paths['biz'], category_list=categories_lst, out_path=paths['out_biz'],
                                     save_ids_path=paths['out_biz_id'], arg_dict=kwarg_dict)
    else:
        biz_id_lst = None

    # check have required info for filtering ops
    for out_arg in ['out_review', 'out_tips', 'out_chkin', 'out_photo']:
        if paths[out_arg] is not None:
            if biz_id_lst is None:
                required_arg_error(parser, ['-b/--biz', '-bi/--biz_ids'])
            else:
                break

    # filter reviews
    if paths['out_review'] is not None:
        if 'dataframe' not in kwarg_dict:
            kwarg_dict['dataframe'] = Df.DASK
        get_reviews(paths['review'], biz_id_lst, paths['out_review'], paths['out_prefilter_review'],
                    arg_dict=kwarg_dict)

    # filter tips
    if paths['out_tips'] is not None:
        get_tips(paths['tips'], biz_id_lst, paths['out_tips'], arg_dict=kwarg_dict)

    # filter checkin
    if paths['out_chkin'] is not None:
        get_checkin(paths['chkin'], biz_id_lst, paths['out_chkin'], arg_dict=kwarg_dict)

    # filter photo
    if paths['out_photo'] is not None:
        get_photos(paths['pin'], biz_id_lst, paths['out_photo'], arg_dict=kwarg_dict)

    # photo dataset
    if paths['out_photo_set'] is not None:
        generate_photo_set(paths['biz_photo'], paths['pin'], biz_id_lst, paths['photo_folder'], paths['out_photo_set'],
                           save_ids_path=paths['out_biz_id'], arg_dict=kwarg_dict)

    # resize photos
    if paths['photo_set_in'] is not None:
        resize_photo_set(paths['photo_set_in'], paths['photo_folder'], arg_dict=kwarg_dict)
