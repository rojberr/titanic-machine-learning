import pickle
import model
import argparse
import pandas as pd

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ID3 tree")
    parser.add_argument("--data_file", type=str,
                        help="<datafile_path> f.e. ../predict.csv", required=True)
    parser.add_argument("--model_file", type=str,
                        help="<model_pickle_path> f.e. ./model.pkl", required=True)
    parser.add_argument("--output_file", type=str,
                        help="<output_path> f.e. ./output.csv", required=True)
    args = parser.parse_args()

    with open(args.model_file, 'rb') as file:
        id3_tree: model.DecisionTreeClassifier = pickle.load(file)
    input_df = pd.read_csv(args.data_file)

    output_df = id3_tree.predict(input_df)
    print(output_df)

    output_df.to_csv(args.output_file, index=False)
    print(f"[INFO]: This script predicted output based on ID3 tree and saved it in {args.output_file} file.")
