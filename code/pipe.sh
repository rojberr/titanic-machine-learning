python3 train.py \
  --data_file=../data/put-titanic-homework.csv \
  --output_file=./model.pkl \
  --omit_columns=PassengerId,Name \
  --target=Survived

python3 predict.py \
#  --data_file=../data/put-titanic-homework.csv \
#  --input_model=./model.pkl \
#  --output_file=./output.csv \
#  --omit_features=PassengerId,Name,Ticket,Cabin \
#  --target=Survived