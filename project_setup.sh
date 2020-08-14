#!/usr/bin/env bash

base_folder="../project/dataset_test"
#base_folder=""

shopt -s nocasematch
proceed=""
while [ -z "$proceed" ]
do
  echo This script takes a \*long\* time to run. Are you sure you wish to proceed [y/n]?
  read -r proceed
  if [[ $proceed != "y" ]] && [[ $proceed != "n" ]]; then
    proceed=""
  fi
done

if [[ $proceed == "n" ]]; then
  exit
fi

shopt -s nocasematch
while [ -z "$base_folder" ]
do
  echo Enter path folder containing \'yelp_dataset.tar\' and \'yelp_photos.tar\'
  echo \(Press \'enter\' for current folder or \'q\' to quit\)
  read -r base_folder
  if [[ $base_folder == "" ]]; then
    base_folder="."
  fi
done

if [[ $base_folder == "q" ]]; then
  exit
fi

dataset_file="$base_folder/yelp_dataset.tar"
if [[ ! -f "$dataset_file" ]]; then
  echo "$dataset_file does not exist, please verify location"
  exit
fi
photos_file="$base_folder/yelp_photos.tar"
if [[ ! -f "$photos_file" ]]; then
  echo "$photos_file does not exist, please verify location"
  exit
fi

# untar files
echo "Please be patient this will take a while!"
echo "-----------------------------------------"
echo "Untar $dataset_file"
tar -C $base_folder -xvf $dataset_file
echo "Untar photos_file"
tar -C $base_folder -xvf $photos_file

echo
echo Changing cwd to \'$base_folder\'
cwd=$(pwd)
cd $base_folder || exit

# json to csv conversion
if [[ ! -f "json_to_csv_converter.py" ]]; then
  echo "Downloading json_to_csv_converter.py"
  wget https://github.com/ib-da-ncirl/dataset-examples/raw/master/json_to_csv_converter.py
fi

echo
echo "Converting raw json files to csv"
echo "--------------------------------"
python3 json_to_csv_converter.py yelp_academic_dataset_tip.json
python3 json_to_csv_converter.py yelp_academic_dataset_review.json
python3 json_to_csv_converter.py yelp_academic_dataset_user.json
python3 json_to_csv_converter.py -tc yelp_academic_dataset_checkin.json
python3 json_to_csv_converter.py photos.json
python3 json_to_csv_converter.py -r yelp_academic_dataset_business.json

cd $cwd || exit

# Path to category list csv to create
OC_PATH=categories.txt
# Path to business csv file
B_PATH=yelp_academic_dataset_business.csv
# Path to business csv file to create
OB_PATH=business.csv
# Path to business ids file to create
OBI_PATH=business_ids.txt
# Path to review csv file
R_PATH=yelp_academic_dataset_review.csv
# Path to review csv file to create
OR_PATH=reviews.csv
# Path to pre-filter review csv file to create
OPR_PATH=prefilter_review.csv
# Path to tips csv file
T_PATH=yelp_academic_dataset_tip.csv
# Path to tips csv file to create
OT_PATH=tips.csv
# Path to checkin csv file
CI_PATH=yelp_academic_dataset_checkin.csv
# Path to checkin csv file to create
OCI_PATH=checkin.csv
# Path to photo csv file
PI_PATH=photos.csv
# Path to photo folder
PF_PATH=photos
# Path to photo set csv file to create
OPS_PATH=photo_dataset.csv
# Limit number of business ids to read
#export BIZ_LIMIT=-li 10000
BIZ_LIMIT=

echo
echo "Generating analysis csv"
echo "-----------------------"
echo "Generate categories list"
python3 etl.py -d $base_folder -c categories.json -oc $OC_PATH -p restaurants
echo "Generate business_ids list"
python3 etl.py -d $base_folder -cl $OC_PATH -b $B_PATH -obi $OBI_PATH -dx attributes\.HairSpecializesIn.* -ob $OB_PATH $BIZ_LIMIT
echo "Generate reviews"
python3 etl.py -d $base_folder -bi $OBI_PATH -r $R_PATH -or $OR_PATH
echo "Generate reviews (pre-filtered)"
python3 etl.py -d $base_folder -bi $OBI_PATH -opr $OPR_PATH -r $R_PATH -or $OR_PATH
echo "Generate tips"
python3 etl.py -d $base_folder -bi $OBI_PATH -t $T_PATH -ot $OT_PATH
echo "Generate checkin"
python3 etl.py -d $base_folder -bi $OBI_PATH -ci $CI_PATH -oci $OCI_PATH
echo "Generate photo dataset"
python3 etl.py -d $base_folder -bp $B_PATH -pi $PI_PATH -pf $PF_PATH -bi $OBI_PATH -ops $OPS_PATH -mx pin:label=food

echo
echo "Preprocessing photos"
echo "--------------------"
echo "Generate 299x299 px"
python3 etl.py -d $base_folder -pf $PF_PATH -psi $OPS_PATH -pfr photos299 -ps 299
echo "Generate 224x224 px"
python3 etl.py -d $base_folder -pf $PF_PATH -psi $OPS_PATH -pfr photos224 -ps 224
echo "Generate 150x100 px"
python3 etl.py -d $base_folder -pf $PF_PATH -psi $OPS_PATH -pfr photos150_100 -ps 150,100

echo
echo Thanks for your patience
