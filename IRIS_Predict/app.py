from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open('iri.pkl', 'rb') as model_file:
    model = pickle.load(model_file)
predicted = {
    'Iris-setosa': '/static/img/iris-setosa.jpg',
    'Iris-versicolor': '/static/img/iris-versicolor.jpg',
    'Iris-virginica': '/static/img/Iris_virginica.jpg'
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        sepal_length = float(request.form['a'])
        sepal_width = float(request.form['b'])
        petal_length = float(request.form['c'])
        petal_width = float(request.form['d'])


        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])


        prediction = model.predict(features)[0]


        species_mapping = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
        predicted_species = species_mapping[prediction]

        image_path = predicted.get(predicted_species, '')

        return render_template('result.html', prediction_message=f'The predicted species is {predicted_species}.', image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
