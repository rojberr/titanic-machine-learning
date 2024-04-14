import pickle
import model

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

    with open('model.pkl', 'rb') as file:
        id3_tree: model.DecisionTreeClassifier = pickle.load(file)

    print_tree(id3_tree.node)

    #model = joblib.load('model.pkl')

    # Load the test data
    #test_data = pd.read_csv('test.csv')

    # Make a prediction
    #prediction = model.predict(test_data)

    # Save the prediction
    #np.savetxt('prediction.csv', prediction, delimiter=',')
