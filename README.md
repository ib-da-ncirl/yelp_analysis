# yelp_analysis

## etl.py
### Installation
Install dependencies via

    pip3 install -r requirements.txt
    
### Usage
    usage: etl.py [-h] [-d DIR] [-b BIZ | -bi BIZ_IDS] [-r REVIEW] [-t TIPS]
                  [-ci CHKIN] [-pi PIN] [-c CAT | -cl CAT_LIST] [-p PARENT]
                  [-e EXCLUDE] [-ob OUT_BIZ] [-or OUT_REVIEW] [-ot OUT_TIPS]
                  [-oci OUT_CHKIN] [-op OUT_PHOTO] [-bp BIZ_PHOTO]
                  [-pf PHOTO_FOLDER] [-ops OUT_PHOTO_SET] [-oc OUT_CAT]
                  [-obi OUT_BIZ_ID] [-dx DROP_REGEX]
                  [-mx MATCH_REGEX [MATCH_REGEX ...]] [-df {pandas,dask}]
                  [-nr NROWS] [-li LIMIT_ID] [-v]
    
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
      -bp BIZ_PHOTO, --biz_photo BIZ_PHOTO
                            Path to business csv for photo dataset file; absolute
                            or relative to 'root directory' if argument supplied
      -pf PHOTO_FOLDER, --photo_folder PHOTO_FOLDER
                            Path to photo folder; absolute or relative to 'root
                            directory' if argument supplied
      -ops OUT_PHOTO_SET, --out_photo_set OUT_PHOTO_SET
                            Path to photo set folder file; absolute or relative to
                            'root directory' if argument supplied
      -oc OUT_CAT, --out_cat OUT_CAT
                            Path to category list csv to create; absolute or
                            relative to 'root directory' if argument supplied
      -obi OUT_BIZ_ID, --out_biz_id OUT_BIZ_ID
                            Path to business ids csv to create; absolute or
                            relative to 'root directory' if argument supplied
      -dx DROP_REGEX, --drop_regex DROP_REGEX
                            Regex for business csv columns to drop
      -mx MATCH_REGEX [MATCH_REGEX ...], --match_regex MATCH_REGEX [MATCH_REGEX ...]
                            Regex for csv columns to match;
                            'csv_id:column_name=regex'. Valid 'csv_id' are;
                            'biz'=business csv file, 'pin'=photo csv file,
                            'tip'=tip csv file and 'review'=review csv file
      -df {pandas,dask}, --dataframe {pandas,dask}
                            Dataframe to use; 'pandas' or 'dask'
      -nr NROWS, --nrows NROWS
                            Number of rows to read, (Note: ignored with '-df=dask'
                            option)
      -li LIMIT_ID, --limit_id LIMIT_ID
                            Limit number of business ids to read
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
                        /photos/<yelp image files>

##### Generate categories list

    python3 etl.py -d /path/to -c categories.json -oc categories.txt -p restaurants

`categories.json` is read, extracting all the sub-categories of `restaurant` and saves them to `/path/to/categories.txt`. 

##### Generate business ids list

    python3 etl.py -d /path/to -cl categories.txt -b yelp_dataset/yelp_academic_dataset_business.csv -obi business_ids.txt

The list of categories will be read from `categories.txt`, and the ids of all businesses which have one of those categories will be saved to `/path/to/business_ids.txt`.

By default, columns with sub-attributes that have been expanded into individual columns will be dropped.

E.g. {'hours': {'monday': '8-5'}} will be represented by two columns; 'hours' and 'hours.monday'. 

The 'hours' column will be dropped as it is not required.

The `-dx\--drop_regex` option allows the specification of a regular expression, which causes matching columns to be dropped.

E.g. `-dx attributes\.HairSpecializesIn.*` will drop all columns beginning with "attributes.HairSpecializesIn". 

    python3 etl.py -d /path/to -cl categories.txt -b yelp_dataset/yelp_academic_dataset_business.csv -obi business_ids.txt -dx attributes\.HairSpecializesIn.*

The `-li\--limit_id` option sets the maximum number of business ids to retrieve.

E.g. `-li 500` will retrieve the ids of the first 500 businesses.  

    python3 etl.py -d /path/to -cl categories.txt -b yelp_dataset/yelp_academic_dataset_business.csv -obi business_ids.txt -dx attributes\.HairSpecializesIn.* -li 500

##### Generate csv file of businesses filtered by specified business ids 

    python3 etl.py -d /path/to -bi business_ids.txt -r yelp_dataset/yelp_academic_dataset_business.csv -ob business.csv

The list of business ids will be read from `business_ids.txt`, and the contents of `yelp_academic_dataset_business.csv` are filtered to exclude businesses not in the id list. The remaining reviews will be saved to `/path/to/business.csv`.

##### Generate csv file of reviews for specified business ids 

    python3 etl.py -d /path/to -bi business_ids.txt -r yelp_dataset/yelp_academic_dataset_review.csv -or reviews.csv

The list of business ids will be read from `business_ids.txt`, and the contents of `yelp_academic_dataset_business.csv` are filtered to exclude reviews for businesses not in the id list. The remaining reviews will be saved to `/path/to/reviews.csv`.

##### Generate csv file of tips for specified business ids 

    python3 etl.py -d /path/to -bi business_ids.txt -t yelp_dataset/yelp_academic_dataset_tip.csv -ot tips.csv

The list of business ids will be read from `business_ids.txt`, and the contents of `yelp_academic_dataset_tip.csv` are filtered to exclude tips for businesses not in the id list. The remaining tips will be saved to `/path/to/tips.csv`.

##### Generate csv file of photos for specified business ids 

    python3 etl.py -d /path/to -bi business_ids.txt -pi yelp_photos/photos.csv -op biz_photos.csv

The list of business ids will be read from `business_ids.txt`, and the contents of `photos.csv` are filtered to exclude photos for businesses not in the id list. The remaining photos will be saved to `/path/to/biz_photos.csv`.

##### Generate csv file of photos for specified business ids with label 'food' 

    python3 etl.py -d /path/to -bi business_ids.txt -pi yelp_photos/photos.csv -op food_photos.csv -mx pin:label=food

The list of business ids will be read from `business_ids.txt`, and the contents of `photos.csv` are filtered to exclude photos for businesses not in the id list, and of those photos only photos labeled `food` are allowed. The remaining photos will be saved to `/path/to/food_photos.csv`.

The `-mx\--match_regex` allows the specification of a column in a csv file which must match the specified regular expression in order to be included.

E.g. `-mx pin:label=food` requires the `label` column in the photo csv file to match `food`. 

Valid 'csv_id' are:
                                                     
|csv_id|csv file|Comment|
|------|--------|-------|
|biz|business csv file|File specified by *-b/--biz* option|
|pin|photo csv file|File specified by *-pi/--pin* option|
|tip|tips csv file|File specified by *-t/--tips* option|
|review|review csv file|File specified by *-r/--review* option|
                                                     
##### Generate csv file for the photos image classification dataset 

    python3 etl.py -d /path/to -bp yelp_dataset/yelp_academic_dataset_business.csv -pi yelp_photos/photos.csv -pf yelp_photos/photos -bi business_ids.txt -ops photo_dataset.csv -mx pin:label=food

The list of business ids will be read from `business_ids.txt`, and the contents of `yelp_academic_dataset_business.csv` are filtered to exclude photos for businesses not in the id list. Similarly, the contents of `photos.csv` are filtered to exclude photos for businesses not in the id list, and of those photos only photos labeled `food` are allowed. The actual photos from the `/path/to/yelp_photos/photos` folder are analysed, and the resultant dataset is saved to `/path/to/photo_dataset.csv`.

##### Generate csv file of checkins for specified business ids

    python3 etl.py -d /path/to -bi business_ids.txt -ci yelp_dataset/yelp_academic_dataset_checkin.csv -oci checkin.csv

The list of business ids will be read from `business_ids.txt`, and the contents of `yelp_academic_dataset_checkin.csv` are filtered to exclude businesses not in the id list. The remaining checkin counts will be saved to `/path/to/checkin.csv`.

##### Sample use case

Generate the files required for an analysis of restaurants. The required data is:

|Filename|Description|
|--------|-----------|
|categories.txt|List of sub-categories of the 'restaurants' category|
|business_ids.txt|List of ids of businesses matching at least one sub-categories of the 'restaurants' category|  
|business.csv|Business information for businesses with ids listed in business_ids.txt|  
|reviews.csv|Reviews for businesses with ids listed in business_ids.txt|  
|tips.csv|Tips for businesses with ids listed in business_ids.txt|  
|checkin.csv|Checkin counts for businesses with ids listed in business_ids.txt|  
|photo_dataset.csv|Photo dataset for image classification of businesses with ids listed in business_ids.txt|  

###### Step by step

    python3 etl.py -d /path/to -c yelp_dataset/categories.json -oc categories.txt -p restaurants
    python3 etl.py -d /path/to -cl categories.txt -b yelp_dataset/yelp_academic_dataset_business.csv -obi business_ids.txt -dx attributes\.HairSpecializesIn.* -ob business.csv
    python3 etl.py -d /path/to -bi business_ids.txt -r yelp_dataset/yelp_academic_dataset_review.csv -or reviews.csv
    python3 etl.py -d /path/to -bi business_ids.txt -t yelp_dataset/yelp_academic_dataset_tip.csv -ot tips.csv
    python3 etl.py -d /path/to -bi business_ids.txt -ci yelp_dataset/yelp_academic_dataset_checkin.csv -oci checkin.csv
    python3 etl.py -d /path/to -bp yelp_dataset/yelp_academic_dataset_business.csv -pi yelp_photos/photos.csv -pf yelp_photos/photos -bi business_ids.txt -ops photo_dataset.csv -mx pin:label=food

###### All in one

    python3 etl.py -d /path/to -c categories.json -oc categories.txt -p restaurants -b yelp_dataset/yelp_academic_dataset_business.csv -obi business_ids.txt -dx attributes\.HairSpecializesIn.* -ob business.csv -r yelp_dataset/yelp_academic_dataset_review.csv -or reviews.csv -t yelp_dataset/yelp_academic_dataset_tip.csv -ot tips.csv -bp yelp_dataset/yelp_academic_dataset_business.csv -pi yelp_photos/photos.csv -pf yelp_photos/photos -ops photo_dataset.csv -mx pin:label=food



  