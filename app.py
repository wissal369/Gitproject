from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import os
import pickle

app = Flask(__name__)

# Vérifie si le modèle existe déjà
if not os.path.exists('model.pkl'):
    # Entraînement du modèle
    df = pd.read_csv('Social_Network_Ads.csv')
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    X = df[['Gender', 'Age', 'EstimatedSalary']]
    y = df['Purchased']

    model = SVC(probability=True)
    model.fit(X, y)

    # Sauvegarde du modèle
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

# Charger le modèle
with open('model.pkl', 'rb') as f:

    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    gender = request.form['gender']
    age = int(request.form['age'])
    salary = int(request.form['salary'])
    gender_val = 1 if gender == 'Male' else 0

    input_data = np.array([[gender_val, age, salary]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    result = f"✅ L'utilisateur VA acheter (Probabilité : {probability:.2f}%)" if prediction == 1 \
             else f"❌ L'utilisateur NE va PAS acheter (Probabilité : {probability:.2f}%)"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
