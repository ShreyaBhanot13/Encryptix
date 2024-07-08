import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pickle

sns.set(style="white", color_codes=True)

iris = pd.read_csv("IRIS.csv")
print(iris.head())

print(iris["species"].value_counts())

sns.FacetGrid(iris, hue="species", height=6).map(plt.scatter, "petal_length", "sepal_width").add_legend()
plt.show()

flower_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
iris['species'] = iris['species'].map(flower_mapping)

print(iris.head())

X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y = iris['species'].values

model = LogisticRegression()
model.fit(X, y)

with open('iri.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

print(f"Model Accuracy: {model.score(X, y):.2f}")

expected = y
predicted = model.predict(X)


print("\nClassification Report:")
print(metrics.classification_report(expected, predicted))

print("\nConfusion Matrix:")
print(metrics.confusion_matrix(expected, predicted))
