import pickle
import argparse
import pandas as pd
import model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ID3 tree")
    parser.add_argument("--data_file", type=str,
                        help="<datafile_path> f.e. ../data.csv", required=True)
    parser.add_argument("--output_file", type=str,
                        help="<output_pickle_path> f.e. ./model.pkl", required=True)
    parser.add_argument("--omit_columns", type=str,
                        help="<columns_to_omit_divided_by_comma> f.e. Survived,Name,PassengerId", required=True)
    parser.add_argument("--target", type=str,
                        help="<target> f.e. Survived", required=True)
    args = parser.parse_args()

    DATAFILE = args.data_file
    columns_to_omit = args.omit_columns.split(",")

    input_df = pd.read_csv(DATAFILE, usecols=lambda column: column not in columns_to_omit)
    columns_to_omit.append(args.target)
    selected_feature_columns = [col for col in input_df.columns if col not in columns_to_omit]

    titanic_tree = model.DecisionTreeClassifier(
        predictors_X=input_df.values,
        feature_names=selected_feature_columns,
        labels=input_df[args.target].values,
    )
    titanic_tree.id3()

    with open(args.output_file, 'wb') as file:
        pickle.dump(titanic_tree, file)

    print(f"[INFO]: This script trained ID3 tree and saved it in {args.output_file} file as pickle.")
