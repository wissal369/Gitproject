import streamlit as st
import numpy as np
import joblib

# Charger le modèle
model = joblib.load("social_ads_model.pkl")

st.title("Prédiction d'Achat de Produit (Social Network Ads)")

st.markdown("Entrez les informations de l'utilisateur :")

# Inputs utilisateur : seulement les 2 features utilisées pour entraîner le modèle
age = st.slider("Âge", 18, 60, 30)
salary = st.slider("Salaire estimé (€)", 15000, 150000, 50000)

# Création des données d'entrée (exactement 2 colonnes)
input_data = np.array([[age, salary]])

# Prédiction
if st.button("Prédire"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("✅ Cet utilisateur achètera probablement le produit.")
    else:
        st.warning("❌ Cet utilisateur n’achètera probablement pas le produit.")
