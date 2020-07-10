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
    categories = categories_str.split(',')
    for category in categories:
        req = category.strip() in category_list
        if req:
            break
    return req


def duration(start, verbose):
    if verbose:
        print(f"Duration: {timer() - start:.1f}s")


def get_business(business_path, category_list, out_path, save_ids_path, drop_cols=None, verbose=False):
    """
    Process the businesses csv file to get a list of businesses with the specified categories
    :param business_path: Path to business csv file
    :param category_list: List of categories to match
    :param out_path: File to save filtered businesses to
    :param save_ids_path: File to save business id list to
    :param drop_cols: Regex for columns to drop
    :param verbose: Verbose mode flag
    :return:  Business id list
    """
    start = timer()

    # read business details
    # - DtypeWarning: Columns (36,37,41,43,46,56,67,72,76,80,99) have mixed types
    #   most of the columns have bool mixed with missing values so use a converter to preserve the missing value
    #   for the moment
    def bool_or_none(x):
        return bool(x) if x else None

    biz_df = pd.read_csv(business_path, dtype={
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
    })

    # find root columns which are not required as values have been expanded into their own columns
    # e.g. 'hours' is not required as info is in 'hours.Friday' etc.
    # also any columns matching the drop column regex
    col_to_drop = set()
    for col in biz_df.columns.values:
        if "." in col:
            splits = col.split(".")
            col_name = splits[0]
            col_to_drop.add(col_name)
            for col_idx in range(1, len(splits) - 1):
                col_name = col_name + '.' + splits[col_idx]
                col_to_drop.add(col_name)

        if drop_cols is not None:
            if re.search(drop_cols, col):
                col_to_drop.add(col)

    if len(col_to_drop) > 0:
        print(f'Dropping {len(col_to_drop)} unnecessary columns')
        print(f'{col_to_drop}')
        biz_df = biz_df.drop(list(col_to_drop), axis=1)

    biz_df = biz_df.fillna('')

    print(f"{len(biz_df)} businesses loaded")
    print(f"Filtering for businesses of required {len(category_list)} categories")

    # 'categories' column is a string containing a comma separated list of categories
    biz_df['req'] = biz_df['categories'].apply(lambda lst: check_categories(lst, category_list))

    # get just the required businesses
    req_biz_df = biz_df[biz_df['req']]

    req_biz_df = req_biz_df.drop(['req'], axis=1)

    req_biz_id_lst = req_biz_df['business_id'].to_list()

    print(f"{len(req_biz_df)} businesses identified")
    duration(start, verbose)

    if out_path is not None:
        print(f"Saving to '{out_path}'")
        req_biz_df.to_csv(out_path, index=False)

    if save_ids_path is not None:
        save_list(save_ids_path, req_biz_id_lst)

    return req_biz_id_lst


def filter_by_biz(entity_name, csv_path, dtype, id_lst, out_path, verbose=False, df=Df.PANDAS):
    """
    Get the entities for the specified list of business ids
    :param entity_name: Name of entity being processed
    :param csv_path: Path to entity csv file
    :param dtype: dtypes for read_csv
    :param id_lst:  List of business ids
    :param out_path: Path to save filtered entities to
    :param verbose: Verbose mode flag
    :param df: DataFrame to use; Df.PANDAS or Df.DASK
    :return:
    """
    start = timer()

    hash_id_list = []
    for bid in id_lst:
        hash_id_list.append(hash(bid))

    print(f"Reading '{csv_path}'")
    if df == Df.PANDAS:
        rev_df = pd.read_csv(csv_path, dtype=dtype, encoding='utf-8', engine='python', quoting=csv.QUOTE_MINIMAL)
    elif df == Df.DASK:
        rev_df = dd.read_csv(csv_path, dtype=dtype, encoding='utf-8', engine='python', quoting=csv.QUOTE_MINIMAL)
    else:
        raise NotImplemented(f'Unrecognised dataframe type: {df}')

    print(f"{len(rev_df)} {entity_name} loaded")
    print(f"Filtering for {entity_name} of required {len(id_lst)} businesses")

    if df == Df.PANDAS:
        rev_df['req'] = rev_df['business_id'].apply(lambda lst: hash(lst) in hash_id_list)
    elif df == Df.DASK:
        rev_df['req'] = rev_df['business_id'].apply(lambda lst: hash(lst) in hash_id_list, meta=('business_id', 'str'))

    # get just the required businesses
    req_rev_df = rev_df[rev_df['req']]
    req_rev_df = req_rev_df.drop(['req'], axis=1)

    print(f"{len(req_rev_df)} {entity_name} identified")
    duration(start, verbose)

    if out_path is not None:
        print(f"Saving to '{out_path}'")
        req_rev_df.to_csv(out_path, index=False)


def get_reviews(csv_path, id_lst, out_path, verbose):
    """
    Get the reviews for the specified list of business ids
    :param csv_path: Path to review csv file
    :param id_lst:  List of business ids
    :param out_path: Path to save filtered reviews to
    :param verbose: Verbose mode flag
    :return:
    """
    filter_by_biz('reviews', csv_path, {
        "cool": object,
        "funny": object,
        "useful": object,
        "stars": object,
    }, id_lst, out_path, verbose=verbose, df=Df.DASK)


def get_tips(csv_path, id_lst, out_path, verbose):
    """
    Get the tips for the specified list of business ids
    :param csv_path: Path to tip csv file
    :param id_lst:  List of business ids
    :param out_path: Path to save filtered reviews to
    :param verbose: Verbose mode flag
    :return:
    """
    filter_by_biz('tips', csv_path, {
        "compliment_count": object,
    }, id_lst, out_path, verbose=verbose, df=Df.DASK)


def get_checkin(csv_path, id_lst, out_path, verbose):
    """
    Get the checkin for the specified list of business ids
    :param csv_path: Path to checkin csv file
    :param id_lst:  List of business ids
    :param out_path: Path to save filtered reviews to
    :param verbose: Verbose mode flag
    :return:
    """
    filter_by_biz('checkins', csv_path, {}, id_lst, out_path, verbose=verbose, df=Df.DASK)


def get_photos(csv_path, id_lst, out_path, verbose):
    """
    Get the photos for the specified list of business ids
    :param csv_path: Path to checkin csv file
    :param id_lst:  List of business ids
    :param out_path: Path to save filtered reviews to
    :param verbose: Verbose mode flag
    :return:
    """
    filter_by_biz('photos', csv_path, {}, id_lst, out_path, verbose=verbose, df=Df.PANDAS)


def file_arg_help(descrip, action=None):
    if action is None:
        tail = ';'
    else:
        tail = f' {action};'
    return f"Path to {descrip} file{tail} absolute or relative to 'root directory' if argument supplied"


def warning(msg):
    print(f"Warning: {msg}")


def ignore_arg_warning(args_namespace, arg_lst):
    for arg in arg_lst:
        if arg in args_namespace:
            warning(f"Ignoring '{arg}' argument")


def required_arg_error(arg_parser, req_arg):
    arg_parser.print_usage()
    sys.exit(f"{os.path.split(sys.argv[0])[1]}: error one of the arguments {req_arg} is required")


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
    # review, tips, checkin & photo
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
    for arg_tuple in [('-oc', '--out_cat', 'category list'),
                      ('-obi', '--out_biz_id', 'business ids')]:
        parser.add_argument(
            arg_tuple[0], arg_tuple[1], type=str, help=file_arg_help(f"{arg_tuple[2]} csv", "to create"),
            default=argparse.SUPPRESS, required=False
        )
    parser.add_argument('-dx', '--drop_regex', help='Regex for business csv columns to drop', type=str, default=None)
    parser.add_argument('-v', '--verbose', help='Verbose mode', action='store_true')

    # TODO add pandas or dask choice in arguments

    args = parser.parse_args()

    if args.verbose:
        print(f"Arguments: {args}")

    paths = {'biz': None, 'biz_ids': None,
             'cat': None, 'cat_list': None,
             'review': None, 'tips': None, 'chkin': None, 'pin': None,
             'out_biz': None, 'out_review': None, 'out_tips': None, 'out_chkin': None, 'out_photo': None,
             'out_cat': None, 'out_biz_id': None}
    for arg_tuple in [('biz', args.biz), ('biz_ids', args.biz_ids),
                      ('cat', args.cat), ('cat_list', args.cat_list)]:
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
                      ('out_cat', None if 'out_cat' not in args else args.out_cat),
                      ('out_biz_id', None if 'out_biz_id' not in args else args.out_biz_id)]:
        if arg_tuple[0] in args:
            paths[arg_tuple[0]] = arg_tuple[1]
    if len(args.dir) > 0:
        for key, val in paths.items():
            if val is not None and not os.path.isabs(val):
                paths[key] = os.path.join(args.dir, val)

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
        biz_id_lst = get_business(paths['biz'], categories_lst, paths['out_biz'], paths['out_biz_id'],
                                  drop_cols=args.drop_regex, verbose=args.verbose)
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
        get_reviews(paths['review'], biz_id_lst, paths['out_review'], args.verbose)

    # filter tips
    if paths['out_tips'] is not None:
        get_tips(paths['tips'], biz_id_lst, paths['out_tips'], args.verbose)

    # filter checkin
    if paths['out_chkin'] is not None:
        # TODO handle long lines in raw checkin data
        raise NotImplemented('Currently not supported, raw file requires preprocessing for long lines')
        #get_checkin(paths['chkin'], biz_id_lst, paths['out_chkin'], args.verbose)

    # filter photo
    if paths['out_photo'] is not None:
        # TODO add filtering for photo label option
        get_photos(paths['pin'], biz_id_lst, paths['out_photo'], args.verbose)
