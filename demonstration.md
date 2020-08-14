# yelp_analysis demonstration
The following steps outline the procedure to set up the project.

## Setup from scratch

***Please note as some files are large, it will take time to complete this entire procedure, possibly up to over 2 hours***

1. Download the following files to an empty folder:
    - yelp_dataset.tar
    - yelp_photos.tar
    - categories.json
1. Run the script project_setup.sh, and provide the path to the folder containing the files above.

    This will:
    - extract the tar files
    - download `json_to_csv_converter.py`, the json to csv converter script if required
    - convert the raw json files to csv files
    - generate the required dataset files for analysis

## Functionality Demonstration
Due to the large dataset size it is recommended to run the following demonstrations on a small subset of the data.
Please see the [photo_stars usage](README.md#usage-1) for more details of arguments.

- Training

    Runs a training demonstration limited to 8000 photos. 


        python3 photo_stars.py -l 8000 -d training_demo    

- Prediction

    Runs a prediction demonstration limited to 8000 photos. 


        python3 photo_stars.py -l 8000 -d prediction_demo

- Tuning

    Runs a tuning demonstration limited to 8000 photos. 
    
    ***Note: this demonstration takes a long time due to the number of parameters to be tuned*** 


        python3 photo_stars.py -l 8000 -d tuning_demo    
  