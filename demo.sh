#!/bin/bash

if [[ ! -d "../../processed_data/photos150_100" ]]; then
  if [[ -f "../../processed_data/photos150_100.zip" ]]; then
    echo 'photos150_100.zip' located
    proceed=""
    while [ -z "$proceed" ]
    do
      echo 'photos150_100.zip' needs to be uncompressed to continue. Are you sure you wish to proceed [y/n]?
      read -r proceed
      if [[ $proceed != "y" ]] && [[ $proceed != "n" ]]; then
        proceed=""
      fi
    done

    if [[ $proceed == "n" ]]; then
      exit
    fi
    unzip "../../processed_data/photos150_100.zip" -d "../../processed_data"
  else
    echo "Unable to locate 'photos150_100'"
    exit
  fi
fi


PS3='Please enter your choice: '
options=("Training demo" "Prediction demo" "Tuning demo" "Quit")
select opt in "${options[@]}"
do
    case $opt in
        "Training demo")
            python3 photo_stars.py -l 8000 -d training_demo
            ;;
        "Prediction demo")
            python3 photo_stars.py -l 8000 -d prediction_demo
            ;;
        "Tuning demo")
            python3 photo_stars.py -l 8000 -d tuning_demo
            ;;
        "Quit")
            break
            ;;
        *) echo "invalid option $REPLY";;
    esac
done