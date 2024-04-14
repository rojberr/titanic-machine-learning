import pickle
import model

if __name__ == '__main__':

    with open('model.pkl', 'rb') as file:
        id3_tree: model.DecisionTreeClassifier = pickle.load(file)

    print(id3_tree)

    #model = joblib.load('model.pkl')

    # Load the test data
    #test_data = pd.read_csv('test.csv')

    # Make a prediction
    #prediction = model.predict(test_data)

    # Save the prediction
    #np.savetxt('prediction.csv', prediction, delimiter=',')
