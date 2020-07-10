# yelp_analysis

## etl.py
### Usage
    usage: etl.py [-h] [-d DIR] (-b BIZ | -bi BIZ_IDS) [-r REVIEW] [-t TIPS]
                  [-ci CHKIN] [-pi PIN] (-c CAT | -cl CAT_LIST) [-p PARENT]
                  [-e EXCLUDE] [-ob OUT_BIZ] [-or OUT_REVIEW] [-ot OUT_TIPS]
                  [-oci OUT_CHKIN] [-op OUT_PHOTO] [-oc OUT_CAT] [-obi OUT_BIZ_ID]
                  [-v]
    
    Perform ETL on the Yelp Dataset CSV data to extract the subset of
    businesses/reviews etc. based on a parent category
    
    optional arguments:
      -h, --help            show this help message and exit
      -d DIR, --dir DIR     Root directory
      -b BIZ, --biz BIZ     Path to business csv file; absolute or relative to
                            'root directory' if argument supplied
      -bi BIZ_IDS, --biz_ids BIZ_IDS
                            Path to business ids file; absolute or relative to
                            'root directory' if argument supplied
      -r REVIEW, --review REVIEW
                            Path to review csv file; absolute or relative to 'root
                            directory' if argument supplied
      -t TIPS, --tips TIPS  Path to tips csv file; absolute or relative to 'root
                            directory' if argument supplied
      -ci CHKIN, --chkin CHKIN
                            Path to checkin csv file; absolute or relative to
                            'root directory' if argument supplied
      -pi PIN, --pin PIN    Path to photo csv file; absolute or relative to 'root
                            directory' if argument supplied
      -c CAT, --cat CAT     Path to categories json file; absolute or relative to
                            'root directory' if argument supplied
      -cl CAT_LIST, --cat_list CAT_LIST
                            Path to category list file; absolute or relative to
                            'root directory' if argument supplied
      -p PARENT, --parent PARENT
                            Parent category
      -e EXCLUDE, --exclude EXCLUDE
                            Exclude categories; a comma separated list of
                            categories to exclude
      -ob OUT_BIZ, --out_biz OUT_BIZ
                            Path to business csv file; absolute or relative to
                            'root directory' if argument supplied
      -or OUT_REVIEW, --out_review OUT_REVIEW
                            Path to review csv file; absolute or relative to 'root
                            directory' if argument supplied
      -ot OUT_TIPS, --out_tips OUT_TIPS
                            Path to tips csv file; absolute or relative to 'root
                            directory' if argument supplied
      -oci OUT_CHKIN, --out_chkin OUT_CHKIN
                            Path to checkin csv file; absolute or relative to
                            'root directory' if argument supplied
      -op OUT_PHOTO, --out_photo OUT_PHOTO
                            Path to photo csv file; absolute or relative to 'root
                            directory' if argument supplied
      -oc OUT_CAT, --out_cat OUT_CAT
                            Path to category list csv file to create; absolute or
                            relative to 'root directory' if argument supplied
      -obi OUT_BIZ_ID, --out_biz_id OUT_BIZ_ID
                            Path to business ids csv file to create; absolute or
                            relative to 'root directory' if argument supplied
      -dx DROP_REGEX, --drop_regex DROP_REGEX
                            Regex for business csv columns to drop
      -v, --verbose         Verbose mode


#### Example command lines

The following examples, assume the directory structure:

    /path/to/yelp_dataset/categories.json
    /path/to/yelp_dataset/yelp_academic_dataset_business.csv
                         /yelp_academic_dataset_checkin.csv
                         /yelp_academic_dataset_review.csv
                         /yelp_academic_dataset_tip.csv
                         /yelp_academic_dataset_user.csv
    /path/to/yelp_photos/photos.csv

##### Generate categories list

    python -d /path/to -c yelp_dataset/categories.json -oc categories.txt -p restaurant

`categories.json` is read, extracting all the sub-categories of `restaurant` and saves them to `/path/to/categories.txt`. 

##### Generate business ids list

    python -d /path/to -cl categories.txt -b yelp_dataset/yelp_academic_dataset_business.csv -obi business_ids.txt

The list of categories will be read from `categories.txt`, and the ids of all businesses which have one of those categories will be saved to `/path/to/business_ids.txt`.

By default, columns with sub-attributes that have been expanded into individual columns will be dropped.

E.g. {'hours': {'monday': '8-5'}} will be represented by two columns; 'hours' and 'hours.monday'. 

The 'hours' column will be dropped as it is not required.
The `-dx\--drop_regex` option allows the specification of a regular expression, which causes matching columns to be dropped.

E.g. `-dx attributes\.HairSpecializesIn.*` will drop all columns beginning with "attributes.HairSpecializesIn". 

    python -d /path/to -cl categories.txt -b yelp_dataset/yelp_academic_dataset_business.csv -obi business_ids.txt -dx attributes\.HairSpecializesIn.*

##### Generate csv file of businesses filtered by specified business ids 

    python -d /path/to -bi business_ids.txt -r yelp_dataset/yelp_academic_dataset_business.csv -ob business.csv

The list of business ids will be read from `business_ids.txt`, and the contents of `yelp_academic_dataset_business.csv` are filtered to exclude businesses not in the id list. The remaining reviews will be saved to `/path/to/business.csv`.

##### Generate csv file of reviews for specified business ids 

    python -d /path/to -bi business_ids.txt -r yelp_dataset/yelp_academic_dataset_review.csv -or reviews.csv

The list of business ids will be read from `business_ids.txt`, and the contents of `yelp_academic_dataset_business.csv` are filtered to exclude reviews for businesses not in the id list. The remaining reviews will be saved to `/path/to/reviews.csv`.

##### Generate csv file of tips for specified business ids 

    python -d /path/to -bi business_ids.txt -t yelp_dataset/yelp_academic_dataset_tip.csv -ot tips.csv

The list of business ids will be read from `business_ids.txt`, and the contents of `yelp_academic_dataset_tip.csv` are filtered to exclude tips for businesses not in the id list. The remaining tips will be saved to `/path/to/tips.csv`.

##### Generate csv file of photos for specified business ids 

    python -d /path/to -bi business_ids.txt -pi yelp_photos/photos.csv -op biz_photos.csv

The list of business ids will be read from `business_ids.txt`, and the contents of `photos.csv` are filtered to exclude photos for businesses not in the id list. The remaining photos will be saved to `/path/to/biz_photos.csv`.

##### Generate csv file of checkins for specified business ids

As some line lengths exceed the max allowed, this functionality is currently unsupported. 

##### Do all at once

    python -d /path/to -c yelp_dataset/categories.json -oc categories.txt -p restaurant -b yelp_dataset/yelp_academic_dataset_business.csv -obi business_ids.txt -ob business.csv  -r yelp_dataset/yelp_academic_dataset_review.csv -or reviews.csv -t yelp_dataset/yelp_academic_dataset_tip.csv -ot tips.csv -pi yelp_photos/photos.csv -op biz_photos.csv



  