import pickle
import model
import argparse
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ID3 tree")
    parser.add_argument("--test_data", type=str,
                        help="<datafile_path> f.e. ../test.csv", required=True)
    parser.add_argument("--model_file", type=str,
                        help="<model_pickle_path> f.e. ./model.pkl", required=True)
    parser.add_argument("--output_file", type=str,
                        help="<output_path> f.e. ./test-output.csv", required=True)
    args = parser.parse_args()

    with open(args.model_file, 'rb') as file:
        id3_tree: model.DecisionTreeClassifier = pickle.load(file)
    input_df = pd.read_csv(args.test_data)

    output_df = id3_tree.predict(input_df)
    print(output_df)

    output_df.to_csv(args.output_file, index=False)

    # Check last 2 columns
    last_two_columns = output_df.iloc[:, -2:]
    equal_values = (last_two_columns.iloc[:, 0] == last_two_columns.iloc[:, 1]).sum()

    # Calculate percentage
    total_values = len(output_df)
    percentage_equal = (equal_values / total_values) * 100

    print(f"The percentage of equal values in the last two columns is: {percentage_equal:.2f}%")


    print(f"[INFO]: This script tested predictions output based on ID3 tree and saved it in {args.output_file} file.")
