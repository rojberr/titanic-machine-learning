import pickle
from contextlib import redirect_stdout

import model
import argparse


def print_tree(node, depth=1):
    if node is None:
        return
    print("| " * depth + " └ " + str(node.value))
    # Recursively print each child
    if node.childs is not None:
        for child in node.childs:
            print_tree(child, depth + 1)
    elif node.next is not None:
        print_tree(node.next, depth + 1)
    else:
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train ID3 tree")
    parser.add_argument("--model_file", type=str,
                        help="<model_pickle_path> f.e. ./model.pkl", required=True)
    parser.add_argument("--output_plot_file", type=str,
                        help="<output_plot_path> f.e. ./output_plot.jpg", required=True)
    args = parser.parse_args()

    with open(args.model_file, 'rb') as file:
        id3_tree: model.DecisionTreeClassifier = pickle.load(file)

    print_tree(id3_tree.node)

    with open(args.output_plot_file, 'w') as file:
        with redirect_stdout(file):
            print_tree(id3_tree.node)

    print(f"[INFO]: This script plotted ID3 tree model and saved it in {args.output_plot_file} file.")
