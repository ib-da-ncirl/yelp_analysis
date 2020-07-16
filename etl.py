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
import pandas as pd
import dask.dataframe as dd
import argparse
from timeit import default_timer as timer
from enum import Enum
from PIL import Image

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


def verify_path(path, typ='file'):
    exists = os.path.exists(path)
    if typ == 'folder':
        valid = os.path.isdir(path)
    elif typ == 'file':
        valid = os.path.isfile(path)
    else:
        raise ValueError(f"Unreckonised 'typ' argument: {typ}")
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

    col = None
    regex = None
    if 'regex' in arg_dict and csv_id is not None:
        if csv_id in arg_dict['regex']:
            col = arg_dict['regex'][csv_id][0]
            regex = arg_dict['regex'][csv_id][1]

    # hash ids for faster comparison
    hash_id_list = None
    if filter_id_lst is not None and filter_id_column is not None:
        hash_id_list = []
        for bid in filter_id_lst:
            hash_id_list.append(hash(bid))

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
        'engine': 'python',
        'quoting': csv.QUOTE_MINIMAL
    }
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

    print(f"{len(rev_df)} {entity_name} loaded")

    # filter by column value
    cmts = []
    if col is not None and regex is not None:
        if col not in rev_df.columns:
            error(f"Column '{col}' not in dataframe")

        cmts.append(f"'{col}' column")

        def col_val_fxn(mcol):
            return re.search(regex, mcol)
    else:
        def col_val_fxn(mcol):
            return True

    if filter_lambda is not None:
        cmts.append(f"'{filter_lambda_column}' column")
    if hash_id_list is not None:
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
        if filter_lambda_column is not None and filter_lambda is not None:
            # filter on lambda function
            ok = filter_lambda(filter_props[filter_lambda_column])
        else:
            ok = True
        if ok:
            # filter on column value
            ok = col_val_fxn(None if col is None else filter_props[col])
        if ok:
            if hash_id_list is not None:
                # filter on ids
                ok = hash(filter_props[filter_id_column]) in hash_id_list
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

    # get just the required businesses
    req_rev_df = rev_df[rev_df['req']]
    # drop work column
    req_rev_df = req_rev_df.drop(['req'], axis=1)

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


def get_reviews(csv_path, id_lst, out_path, arg_dict=None):
    """
    Get the reviews for the specified list of business ids
    :param csv_path: Path to review csv file
    :param id_lst:  List of business ids
    :param out_path: Path to save filtered reviews to
    :param arg_dict: keyword arguments
    :return:
    """
    return load_csv('reviews', csv_path, dtype={
        "cool": object,
        "funny": object,
        "useful": object,
        "stars": object,
    }, out_path=out_path, filter_id_column='business_id', filter_id_lst=id_lst, arg_dict=arg_dict, csv_id='review')


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
    :param out_path: Path to save filtered reviews to
    :param arg_dict: keyword arguments
    :return:
    """
    return load_csv('photos', csv_path, out_path=out_path, filter_id_column='business_id', filter_id_lst=id_lst,
                    arg_dict=arg_dict, csv_id='pin')


def progress(cmt, total, current, step=100):
    current += 1
    if current % step == 0 or total == current:
        print(f"{cmt}: {current} ({current*100/total:.1f}%)", flush=True, end='\r' if total > current else '\n')
    return current


def generate_photo_set(biz_path, photo_csv_path, id_lst, photo_folder, out_path, arg_dict=None):
    """
    Generate a csv for the photo dataset
    :param biz_path: Path to business csv file
    :param photo_csv_path: Path to photo csv file
    :param id_lst:  List of business ids
    :param photo_folder: Path to folder containing photos
    :param out_path: Path to save dataset to
    :param arg_dict: keyword arguments
    :return:
    """
    biz_df, _ = get_business(biz_path, id_list=id_lst, arg_dict=arg_dict, csv_id='biz_photo')
    photo_df, _, _ = get_photos(photo_csv_path, id_lst, None, arg_dict=arg_dict)

    print(f"Processing photo details")

    extensions = Image.registered_extensions().keys()
    count = 0

    def img_attrib(photo):
        attrib = None
        for ext in extensions:
            filename = f"{photo}{ext}"
            filepath = os.path.join(photo_folder, filename)
            if os.path.isfile(filepath):
                im = Image.open(filepath)
                nonlocal count
                count = progress("Photo", len(photo_df), count)
                attrib = f"{filename},{im.format},{im.size[0]},{im.size[1]},{im.mode}"
                break
        if attrib is None:
            raise ValueError(f"Unmatched photo id {photo}")
        return attrib

    photo_df['photo_attrib'] = photo_df['photo_id'].apply(img_attrib)
    photo_df[['photo_file', 'format', 'width', 'height', 'mode']] = photo_df.apply(
                lambda row: pd.Series(row['photo_attrib'].split(',')), axis=1)

    progress("Photo", len(photo_df), len(photo_df))

    # many-to-one join, i.e many photos to one business id
    biz_photo_df = pd.merge(photo_df, biz_df, on='business_id', how='outer')

    # just keep required columns
    photo_set_df = biz_photo_df[['business_id', 'stars', 'photo_id', 'photo_file', 'format', 'width', 'height', 'mode']]

    total_rows = len(photo_set_df)

    # drop rows with no photo id, no photo no prediction
    photo_set_df = photo_set_df.dropna()

    photo_set_df['width'] = photo_set_df['width'].astype('int32')
    photo_set_df['height'] = photo_set_df['height'].astype('int32')

    if total_rows - len(photo_set_df) > 0:
        print(f"Dropped {total_rows - len(photo_set_df)} businesses with no photos")

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

    save_csv(photo_set_df, out_path, index=False)


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
                      ('-pi', '--pin', 'photo')]:
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
                      ('-or', '--out_review', 'review'),
                      ('-ot', '--out_tips', 'tips'),
                      ('-oci', '--out_chkin', 'checkin'),
                      ('-op', '--out_photo', 'photo')]:
        parser.add_argument(
            arg_tuple[0], arg_tuple[1], type=str, help=file_arg_help(f"{arg_tuple[2]} csv"),
            default=argparse.SUPPRESS, required=False
        )
    parser.add_argument('-bp', '--biz_photo', type=str, help=file_arg_help("business csv for photo dataset"),
                        default='')     # just so as not to conflict with mutually exclusive --biz/--biz_ids arguments
    parser.add_argument('-pf', '--photo_folder', type=str, help=file_arg_help("photo", target='folder'),
                        default='')
    for arg_tuple in [('-ops', '--out_photo_set', 'photo set')]:
        parser.add_argument(
            arg_tuple[0], arg_tuple[1], type=str, help=file_arg_help(f"{arg_tuple[2]} folder"),
            default=argparse.SUPPRESS, required=False
        )
    for arg_tuple in [('-oc', '--out_cat', 'category list'),
                      ('-obi', '--out_biz_id', 'business ids')]:
        parser.add_argument(
            arg_tuple[0], arg_tuple[1], type=str, help=file_arg_help(f"{arg_tuple[2]} csv", "to create"),
            default=argparse.SUPPRESS, required=False
        )
    parser.add_argument('-dx', '--drop_regex', help='Regex for business csv columns to drop', type=str, default=None)
    parser.add_argument('-mx', '--match_regex', help="Regex for csv columns to match; 'csv_id:column_name=regex'. "
                                                     "Valid 'csv_id' are; 'biz'=business csv file, "
                                                     "'pin'=photo csv file, 'tip'=tip csv file and "
                                                     "'review'=review csv file",
                        type=str, default=argparse.SUPPRESS, nargs='+')
    parser.add_argument('-df', '--dataframe', help="Dataframe to use; 'pandas' or 'dask'",
                        choices=['pandas', 'dask'], required=False)
    parser.add_argument('-nr', '--nrows', help="Number of rows to read, (Note: ignored with '-df=dask' option)",
                        type=int, default=argparse.SUPPRESS)
    parser.add_argument('-li', '--limit_id', help="Limit number of business ids to read",
                        type=int, default=argparse.SUPPRESS)
    parser.add_argument('-v', '--verbose', help='Verbose mode', action='store_true')

    args = parser.parse_args()

    if len(sys.argv) == 1:
        # display help message when no args are passed.
        parser.print_help()
        sys.exit(1)

    if args.verbose:
        print(f"Arguments: {args}")

    paths = {'biz': None, 'biz_ids': None, 'biz_photo': None, 'photo_folder': None,
             'cat': None, 'cat_list': None,
             'review': None, 'tips': None, 'chkin': None, 'pin': None,
             'out_biz': None, 'out_review': None, 'out_tips': None, 'out_chkin': None, 'out_photo': None,
             'out_photo_set': None,
             'out_cat': None, 'out_biz_id': None}
    ip_arg_tuples = [('biz', args.biz, 'file'), ('biz_ids', args.biz_ids, 'file'),
                     ('biz_photo', args.biz_photo, 'file'), ('photo_folder', args.photo_folder, 'folder'),
                     ('cat', args.cat, 'file'), ('cat_list', args.cat_list, 'file')]
    for arg_tuple in ip_arg_tuples:
        if arg_tuple[0] in args:
            paths[arg_tuple[0]] = arg_tuple[1]
    for arg_tuple in [('review', args.review),
                      ('tips', args.tips),
                      ('chkin', args.chkin),
                      ('pin', args.pin)]:
        if arg_tuple[0] in args:
            paths[arg_tuple[0]] = arg_tuple[1]
    for arg_tuple in [('out_biz', None if 'out_biz' not in args else args.out_biz),
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
        if len(arg_tuple[1]) > 0:
            verify_path(paths[arg_tuple[0]], typ=arg_tuple[2])

    kwarg_dict = {'verbose': args.verbose, 'drop_regex': args.drop_regex}
    if args.dataframe is not None:
        kwarg_dict['dataframe'] = Df.PANDAS if args.dataframe == 'pandas' else Df.DASK
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
        get_reviews(paths['review'], biz_id_lst, paths['out_review'], arg_dict=kwarg_dict)

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
                           arg_dict=kwarg_dict)
