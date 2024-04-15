import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ID3 tree")
    parser.add_argument("--data_file", type=str,
                        help="<datafile_path> f.e. ../data/put-titanic-homework.csv", required=True)
    parser.add_argument("--output_csv", type=str,
                        help="<output_preprocessed_csv_path> f.e. ./data.csv", required=True)
    args = parser.parse_args()

    FILE = args.data_file
    input_df = pd.read_csv(FILE)

    input_df['Age'] = input_df['Age'].map(
        lambda x: 'young' if 0 <= x <= 20 else ('middle' if 20 < x <= 40 else 'old'))

    input_df.to_csv('data.csv', index=False)
