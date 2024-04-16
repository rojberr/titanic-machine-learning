from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__)

## Load model
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/', methods=["GET"])
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    final_features_dataframe = pd.DataFrame([features], columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch'])
    prediction = model.predict(final_features_dataframe)

    if prediction['Predicted'].values[0] == 0:
        prediction = 'Not Survived'
    else:
        prediction = 'Survived'

    return render_template('index.html', prediction_text='Prediction: {}'.format(prediction))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
