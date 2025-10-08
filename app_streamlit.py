import streamlit as st
import requests
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Optional

API_URL = "http://127.0.0.1:8000/predict"
ARTIFACTS_DIR = "artifacts"

@st.cache_data(show_spinner=True)
def load_local_artifacts() -> Tuple:
    model = joblib.load(f"{ARTIFACTS_DIR}/best_model.joblib")
    scaler = joblib.load(f"{ARTIFACTS_DIR}/scaler.joblib")
    le_sex = joblib.load(f"{ARTIFACTS_DIR}/label_encoder_sex.joblib")
    le_embarked = joblib.load(f"{ARTIFACTS_DIR}/label_encoder_embarked.joblib")
    feature_names = joblib.load(f"{ARTIFACTS_DIR}/feature_names.joblib")
    return model, scaler, le_sex, le_embarked, feature_names

def predict_local(data: dict, model, scaler, le_sex, le_embarked, feature_names) -> Tuple[int, float]:
    try:
        row = {
            "pclass": data["pclass"],
            "sex": le_sex.transform([data["sex"]])[0],
            "age": data["age"],
            "sibsp": data["sibsp"],
            "parch": data["parch"],
            "fare": data["fare"],
            "embarked": le_embarked.transform([data["embarked"]])[0],
        }
        X = pd.DataFrame([row], columns=feature_names)
        X_scaled = scaler.transform(X)

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_scaled)[:, 1][0]
        else:
            proba = float(model.predict(X_scaled)[0])
        pred = 1 if proba >= 0.5 else 0
        return pred, proba
    except Exception as e:
        st.error(f"Erreur lors de la prédiction locale : {e}")
        return -1, 0.0

def predict_api(data: dict) -> Tuple[Optional[int], Optional[float]]:
    try:
        resp = requests.post(API_URL, json=data)
        resp.raise_for_status()
        json_data = resp.json()
        return json_data.get("prediction"), json_data.get("probability_survival")
    except requests.RequestException as e:
        st.error(f"Erreur de communication avec API : {e}")
        return None, None

def main():

    st.set_page_config(page_title="Titanic Survival Prediction", layout="centered")

    st.title("🚢 PREDICTION DE SURVIE: Titanic")

    tabs = st.tabs(["Prédiction", "À propos"])

    with tabs[0]:
        local_mode = st.checkbox("Mode local (sans appel API)", value=False)

        with st.form(key="passenger_form"):
            pclass = st.selectbox("Classe du passager (pclass)", [1, 2, 3], index=0)
            sex = st.selectbox("Sexe", ["male", "female"], index=1)
            age = st.number_input("Âge", min_value=0.0, max_value=120.0, value=30.0, step=0.5)
            sibsp = st.number_input("Nb frères/soeurs/conjoints à bord", min_value=0, max_value=10, value=0)
            parch = st.number_input("Nb parents/enfants à bord", min_value=0, max_value=10, value=0)
            fare = st.number_input("Tarif payé (€)", min_value=0.0, max_value=1000.0, value=50.0, step=0.1)
            embarked = st.selectbox("Port d'embarquement", ["S", "C", "Q"], index=0)
            submit = st.form_submit_button("Prédire")

        if submit:
            data = {
                "pclass": pclass,
                "sex": sex,
                "age": age,
                "sibsp": sibsp,
                "parch": parch,
                "fare": fare,
                "embarked": embarked,
            }

            if local_mode:
                with st.spinner("Chargement du modèle local et prédiction en cours..."):
                    model, scaler, le_sex, le_embarked, feature_names = load_local_artifacts()
                    pred, proba = predict_local(data, model, scaler, le_sex, le_embarked, feature_names)
            else:
                with st.spinner("Appel à l'API FastAPI en cours..."):
                    pred, proba = predict_api(data)

            if pred is not None and pred != -1:
                col1, col2 = st.columns([1, 3])
                with col1:
                    emoji = "✅" if pred == 1 else "❌"
                    st.markdown(f"<h1 style='font-size:72px'>{emoji}</h1>", unsafe_allow_html=True)

                with col2:
                    survival_text = "Survécu" if pred == 1 else "Décédé"
                    st.markdown(f"## Prédiction : **{survival_text}**")
                    st.progress(int(proba * 100))
                    st.markdown(f"### Probabilité de survie : {proba:.2%}")
            else:
                st.error("Prédiction impossible. Veuillez vérifier vos données ou l'état du serveur.")

    with tabs[1]:
        st.header("À propos")
        st.markdown(
            """
            Cette application a été développée pour prédire la survie d'un passager du Titanic \
            à partir de ses caractéristiques. Elle utilise un modèle de machine learning déployé \
            via une API FastAPI ou localement avec joblib.
            
            - Utilisez l’onglet 'Prédiction' pour tester le modèle.
            - Cochez 'Mode local' pour prédire sans appel réseau.
            """
        )

if __name__ == "__main__":
    main()
