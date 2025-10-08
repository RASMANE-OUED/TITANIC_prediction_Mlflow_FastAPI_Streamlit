"""
ÉTAPE 1: Entraînement du modèle avec MLflow
JeuDeDonnées: Titanic (classification binaire - survie)

Installation requise:
pip install pandas numpy scikit-learn mlflow joblib matplotlib seaborn
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn
import joblib
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any

# =========================
# Configuration globale & graine aléatoire
# =========================
GRAINE_ALEATOIRE = 42
np.random.seed(GRAINE_ALEATOIRE)

# Dossier de sortie (modèles, figures)
DOSSIER_ARTEFACTS = Path("artefacts")
DOSSIER_ARTEFACTS.mkdir(exist_ok=True)

def _prediction_probabilite_securisee(modele, X):
    """Retourne les probabilités si possible; sinon une normalisation de decision_function."""
    if hasattr(modele, "predict_proba"):
        probabilites = modele.predict_proba(X)
        if probabilites.shape[1] == 2:
            return probabilites[:, 1]
        classes_ = getattr(modele, "classes_", np.array([0, 1]))
        if 1 in classes_:
            idx = int(np.where(classes_ == 1)[0][0])
            return probabilites[:, idx]
        return probabilites.max(axis=1)
    if hasattr(modele, "decision_function"):
        df = modele.decision_function(X)
        df = (df - df.min()) / (df.max() - df.min() + 1e-9)
        return df
    return modele.predict(X)

def charger_et_preparer_donnees() -> Tuple[pd.DataFrame, pd.Series, LabelEncoder, LabelEncoder]:
    """
    Charge et prépare les données Titanic.
    Retourne: (X, y, encodeur_sexe, encodeur_embarquement)
    """
    print("📥 Chargement des données Titanic...")

    try:
        import seaborn as sns
        dataframe = sns.load_dataset('titanic')
    except Exception:
        dataframe = pd.read_csv("Titanic.csv")
        dataframe.columns = dataframe.columns.str.lower()
        dataframe.rename(columns={
            'survived': 'survived',
            'pclass': 'pclass',
            'sibsp': 'sibsp',
            'parch': 'parch',
            'fare': 'fare',
            'embarked': 'embarked',
            'sex': 'sex',
            'age': 'age'
        }, inplace=True)

    print(f"✅ Données chargées: {dataframe.shape[0]} lignes, {dataframe.shape[1]} colonnes")

    caracteristiques_utilisees = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
    cible = 'survived'

    dataframe = dataframe[caracteristiques_utilisees + [cible]].dropna(subset=[cible]).copy()

    dataframe['age'] = dataframe['age'].fillna(dataframe['age'].median())
    dataframe['fare'] = dataframe['fare'].fillna(dataframe['fare'].median())
    dataframe['embarked'] = dataframe['embarked'].fillna(dataframe['embarked'].mode()[0])

    encodeur_sexe = LabelEncoder()
    dataframe['sex'] = encodeur_sexe.fit_transform(dataframe['sex'])

    encodeur_embarquement = LabelEncoder()
    dataframe['embarked'] = encodeur_embarquement.fit_transform(dataframe['embarked'])

    X = dataframe[caracteristiques_utilisees]
    y = dataframe[cible].astype(int)

    print(f"✅ Caractéristiques: {list(X.columns)}")
    print(f"✅ Distribution cible: Survécu={int((y==1).sum())}, Décédé={int((y==0).sum())}")

    return X, y, encodeur_sexe, encodeur_embarquement


def tracer_et_sauvegarder_matrice_confusion(y_reel, y_prediction, titre: str, chemin_sortie: Path):
    matrice_confusion = confusion_matrix(y_reel, y_prediction)
    figure = plt.figure(figsize=(4, 4))
    plt.imshow(matrice_confusion, interpolation='nearest')
    plt.title(titre)
    plt.colorbar()
    positions_ticks = np.arange(2)
    plt.xticks(positions_ticks, ['0', '1'])
    plt.yticks(positions_ticks, ['0', '1'])
    seuil = matrice_confusion.max() / 2.0
    for i, j in np.ndindex(matrice_confusion.shape):
        plt.text(j, i, matrice_confusion[i, j],
                 horizontalalignment="center",
                 color="white" if matrice_confusion[i, j] > seuil else "black")
    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    plt.tight_layout()
    figure.savefig(chemin_sortie, bbox_inches='tight')
    plt.close(figure)


def tracer_et_sauvegarder_courbe_roc(y_reel, y_probabilites, titre: str, chemin_sortie: Path):
    taux_faux_positifs, taux_vrais_positifs, _ = roc_curve(y_reel, y_probabilites)
    aire_sous_courbe = roc_auc_score(y_reel, y_probabilites)
    figure = plt.figure(figsize=(5, 4))
    plt.plot(taux_faux_positifs, taux_vrais_positifs, label=f"AUC = {aire_sous_courbe:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.title(titre)
    plt.xlabel("Taux Faux Positifs")
    plt.ylabel("Taux Vrais Positifs")
    plt.legend()
    plt.tight_layout()
    figure.savefig(chemin_sortie, bbox_inches='tight')
    plt.close(figure)


def entrainer_et_logger_modele(
    X_entrainement, X_test, y_entrainement, y_test,
    nom_modele: str, modele, parametres: Dict[str, Any], nom_execution: str
):
    print(f"\n🚀 Entraînement: {nom_execution}")
    print(f"   Modèle: {nom_modele}")
    print(f"   Paramètres: {parametres}")

    with mlflow.start_run(run_name=nom_execution):
        mlflow.log_params(parametres)
        mlflow.log_param("type_modele", nom_modele)

        modele.fit(X_entrainement, y_entrainement)

        y_prediction_entrainement = modele.predict(X_entrainement)
        y_prediction_test = modele.predict(X_test)
        y_probabilites_test = _prediction_probabilite_securisee(modele, X_test)

        metriques = {
            'precision_entrainement': accuracy_score(y_entrainement, y_prediction_entrainement),
            'precision_test': accuracy_score(y_test, y_prediction_test),
            'precision_test': precision_score(y_test, y_prediction_test),
            'rappel_test': recall_score(y_test, y_prediction_test),
            'score_f1_test': f1_score(y_test, y_prediction_test),
        }
        try:
            metriques['auc_roc_test'] = roc_auc_score(y_test, y_probabilites_test)
        except Exception:
            metriques['auc_roc_test'] = float('nan')

        mlflow.log_metrics(metriques)

        horodatage = datetime.now().strftime("%Y%m%d_%H%M%S")
        chemin_matrice_confusion = DOSSIER_ARTEFACTS / f"{nom_execution}_matrice_confusion_{horodatage}.png"
        chemin_courbe_roc = DOSSIER_ARTEFACTS / f"{nom_execution}_courbe_roc_{horodatage}.png"

        try:
            tracer_et_sauvegarder_matrice_confusion(y_test, y_prediction_test, f"MC - {nom_execution}", chemin_matrice_confusion)
            mlflow.log_artifact(str(chemin_matrice_confusion))
        except Exception as e:
            print(f"   (avertissement) Matrice confusion non loggée: {e}")

        try:
            if not np.isnan(metriques['auc_roc_test']):
                tracer_et_sauvegarder_courbe_roc(y_test, y_probabilites_test, f"ROC - {nom_execution}", chemin_courbe_roc)
                mlflow.log_artifact(str(chemin_courbe_roc))
        except Exception as e:
            print(f"   (avertissement) Courbe ROC non loggée: {e}")

        mlflow.sklearn.log_model(modele, "modele")

        print(f"   ✅ Précision Entrainement: {metriques['precision_entrainement']:.4f}")
        print(f"   ✅ Précision Test: {metriques['precision_test']:.4f}")
        print(f"   ✅ Score F1 Test: {metriques['score_f1_test']:.4f}")
        print(f"   ✅ AUC ROC Test: {metriques['auc_roc_test']:.4f}")

        return modele, metriques


def principal():
    print("=" * 74)
    print("  PROJET ML: ENTRAÎNEMENT ET SUIVI AVEC MLFLOW")
    print("  JeuDeDonnées: Titanic (Prédiction de survie)")
    print("=" * 74)

    uri_suivi = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    mlflow.set_tracking_uri(uri_suivi)
    mlflow.set_experiment("Prediction_Survie_Titanic")


    X, y, encodeur_sexe, encodeur_embarquement = charger_et_preparer_donnees()

    X_entrainement, X_test, y_entrainement, y_test = train_test_split(
        X, y, test_size=0.2, random_state=GRAINE_ALEATOIRE, stratify=y
    )

    print(f"\n📊 Split des données:")
    print(f"   Entraînement: {X_entrainement.shape[0]} échantillons")
    print(f"   Test: {X_test.shape[0]} échantillons")

    normaliseur = StandardScaler()
    X_entrainement_normalise = normaliseur.fit_transform(X_entrainement)
    X_test_normalise = normaliseur.transform(X_test)

    joblib.dump(normaliseur, DOSSIER_ARTEFACTS / 'normaliseur.joblib')
    joblib.dump(encodeur_sexe, DOSSIER_ARTEFACTS / 'encodeur_etiquette_sexe.joblib')
    joblib.dump(encodeur_embarquement, DOSSIER_ARTEFACTS / 'encodeur_etiquette_embarquement.joblib')
    print("\n💾 Normaliseur & encodeurs sauvegardés dans ./artefacts")

    modele_1 = RandomForestClassifier(random_state=GRAINE_ALEATOIRE)
    parametres_1 = dict(n_estimateurs=100, profondeur_max=None, echantillons_min_split=2, random_state=GRAINE_ALEATOIRE)
    modele_entraine_1, metriques_1 = entrainer_et_logger_modele(
        X_entrainement_normalise, X_test_normalise, y_entrainement, y_test,
        'ForetAleatoire', modele_1, parametres_1, 'Execution_1_ForetAleatoire_ParDefaut'
    )

    modele_2 = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5, random_state=GRAINE_ALEATOIRE
    )
    parametres_2 = dict(n_estimateurs=200, profondeur_max=10, echantillons_min_split=5, random_state=GRAINE_ALEATOIRE)
    modele_entraine_2, metriques_2 = entrainer_et_logger_modele(
        X_entrainement_normalise, X_test_normalise, y_entrainement, y_test,
        'ForetAleatoire', modele_2, parametres_2, 'Execution_2_ForetAleatoire_Optimise'
    )

    modele_3 = LogisticRegression(max_iter=1000, random_state=GRAINE_ALEATOIRE)
    parametres_3 = dict(max_iterations=1000, penalite='l2', C=1.0, random_state=GRAINE_ALEATOIRE)
    modele_entraine_3, metriques_3 = entrainer_et_logger_modele(
        X_entrainement_normalise, X_test_normalise, y_entrainement, y_test,
        'RegressionLogistique', modele_3, parametres_3, 'Execution_3_RegressionLogistique'
    )

    modele_4 = DecisionTreeClassifier(max_depth=8, min_samples_split=10, random_state=GRAINE_ALEATOIRE)
    parametres_4 = dict(profondeur_max=8, echantillons_min_split=10, random_state=GRAINE_ALEATOIRE)
    modele_entraine_4, metriques_4 = entrainer_et_logger_modele(
        X_entrainement_normalise, X_test_normalise, y_entrainement, y_test,
        'ArbreDecision', modele_4, parametres_4, 'Execution_4_ArbreDecision'
    )

    print("\n" + "=" * 74)
    print("  COMPARAISON DES MODÈLES")
    print("=" * 74)

    modeles_comparaison = [
        ('Execution 1 - RF Par Défaut', metriques_1, modele_entraine_1),
        ('Execution 2 - RF Optimisé', metriques_2, modele_entraine_2),
        ('Execution 3 - RegLog', metriques_3, modele_entraine_3),
        ('Execution 4 - ArbreDec', metriques_4, modele_entraine_4),
    ]

    print(f"\n{'Modèle':<25} {'Précision':<12} {'Score-F1':<12} {'AUC-ROC':<12}")
    print("-" * 70)

    meilleur_nom_modele = None
    meilleur_objet_modele = None
    meilleur_score = -1.0

    for nom, metrique, objet in modeles_comparaison:
        precision = metrique.get('precision_test', float('nan'))
        f1 = metrique.get('score_f1_test', float('nan'))
        auc = metrique.get('auc_roc_test', float('nan'))
        print(f"{nom:<25} {precision:<12.4f} {f1:<12.4f} {auc:<12.4f}")
        if f1 > meilleur_score:
            meilleur_score = f1
            meilleur_nom_modele = nom
            meilleur_objet_modele = objet

    print("\n" + "=" * 74)
    print(f"🏆 MEILLEUR MODÈLE: {meilleur_nom_modele}")
    print(f"   Score F1: {meilleur_score:.4f}")
    print("=" * 74)

    chemin_meilleur_modele = DOSSIER_ARTEFACTS / 'meilleur_modele.joblib'
    joblib.dump(meilleur_objet_modele, chemin_meilleur_modele)
    noms_caracteristiques = list(X.columns)
    joblib.dump(noms_caracteristiques, DOSSIER_ARTEFACTS / 'noms_caracteristiques.joblib')

    print("\n💾 Artefacts sauvegardés dans ./artefacts :")
    print(f"   - {chemin_meilleur_modele.name}               (meilleur modèle)")
    print("   - normaliseur.joblib                        (StandardScaler)")
    print("   - encodeur_etiquette_sexe.joblib             (LabelEncoder pour 'sex')")
    print("   - encodeur_etiquette_embarquement.joblib        (LabelEncoder pour 'embarked')")
    print("   - noms_caracteristiques.joblib                 (noms des caractéristiques)")
    print("   - figures ROC/MC par exécution (si dispo)")

    print("\n✅ ENTRAÎNEMENT TERMINÉ !")
    print("\n📊 Pour voir MLflow UI :")
    print("   mlflow ui")
    print("   Ouvrir: http://localhost:5000")

    return meilleur_objet_modele, normaliseur, noms_caracteristiques, (encodeur_sexe, encodeur_embarquement)

def predire_un(exemple: Dict[str, Any], chemin_modele="artefacts/meilleur_modele.joblib",
                chemin_normaliseur="artefacts/normaliseur.joblib",
                chemin_encodeur_sexe="artefacts/encodeur_etiquette_sexe.joblib",
                chemin_encodeur_embarquement="artefacts/encodeur_etiquette_embarquement.joblib",
                chemin_noms_caracteristiques="artefacts/noms_caracteristiques.joblib") -> float:
    """
    Exemple d'inférence sur un seul passager.
    exemple: dict avec clés pclass, sex(str), age, sibsp, parch, fare, embarked(str)
    Retourne probabilité de survie estimée (entre 0 et 1).
    """
    modele = joblib.load(chemin_modele)
    normaliseur = joblib.load(chemin_normaliseur)
    encodeur_sexe = joblib.load(chemin_encodeur_sexe)
    encodeur_embarquement = joblib.load(chemin_encodeur_embarquement)
    noms_caracteristiques = joblib.load(chemin_noms_caracteristiques)

    ligne = {
        "pclass": exemple["pclass"],
        "sex": encodeur_sexe.transform([exemple["sex"]])[0],
        "age": exemple["age"],
        "sibsp": exemple["sibsp"],
        "parch": exemple["parch"],
        "fare": exemple["fare"],
        "embarked": encodeur_embarquement.transform([exemple["embarked"]])[0],
    }
    X = pd.DataFrame([ligne], columns=noms_caracteristiques)
    X_normalise = normaliseur.transform(X)
    probabilite = _prediction_probabilite_securisee(modele, X_normalise)
    if np.ndim(probabilite) == 0:
        return float(probabilite)
    return float(probabilite[0])

if __name__ == "__main__":
    modele, normaliseur, caracteristiques, encodeurs = principal()

    print("\n" + "=" * 74)
    print("  TEST D'INFÉRENCE RAPIDE")
    print("=" * 74)
    demonstration = {
        "pclass": 1,
        "sex": "female",
        "age": 26,
        "sibsp": 0,
        "parch": 0,
        "fare": 80.0,
        "embarked": "S"  # 'S','C','Q'
    }
    try:
        probabilite = predire_un(demonstration)
        print(f"Probabilité de survie (démo): {probabilite:.3f}")
    except Exception as e:
        print(f"(avertissement) Inférence démo impossible: {e}")

    print("\n" + "=" * 74)
    print("  FICHIERS GÉNÉRÉS")
    print("=" * 74)
    print("✓ artefacts/meilleur_modele.joblib")
    print("✓ artefacts/normaliseur.joblib")
    print("✓ artefacts/encodeur_etiquette_sexe.joblib")
    print("✓ artefacts/encodeur_etiquette_embarquement.joblib")
    print("✓ artefacts/noms_caracteristiques.joblib")
    print("✓ mlruns/ (logs MLflow)")
    print("=" * 74)