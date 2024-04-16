#!/bin/bash

set -e # Break on error

cd "$(dirname "$0")" # Set working directory to the directory of the script

python3 preprocess_data.py \
  --data_file=../data/put-titanic-homework.csv \
  --output_csv=./data.csv

python3 train.py \
  --data_file=./data.csv \
  --output_file=./model.pkl \
  --omit_columns=PassengerId,Name \
  --target=Survived

python3 plot.py \
  --model_file=./model.pkl \
  --output_file=./plot.txt

# Test on same data as trained
python3 test.py \
  --test_data=./data.csv \
  --model_file=./model.pkl \
  --output_file=./test_output_1.csv

# Test on kaggle data
python test.py \
  --test_data=../data/kaggle-titanic-train.csv \
  --model_file=./model.pkl \
  --output_file=./test_output_2.csv

python3 predict.py \
  --data_file=./data.csv \
  --model_file=./model.pkl \
  --output_file=./output.csv
