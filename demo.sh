#!/bin/bash

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