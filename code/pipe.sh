#!/bin/bash

set -e # Break on error

python3 data_preprocessing.py \
  --data_file=../data/put-titanic-homework.csv \
  --output_csv=./data.csv

python3 train.py \
  --data_file=./data.csv \
  --output_file=./model.pkl \
  --omit_columns=PassengerId,Name \
  --target=Survived

python3 test.py \
  --test_data=./test-data.csv \
  --model_file=./model.pkl \
  --output_file=./test_output.csv

python3 predict.py \
  --data_file=./data.csv \
  --model_file=./model.pkl \
  --output_file=./output.csv
