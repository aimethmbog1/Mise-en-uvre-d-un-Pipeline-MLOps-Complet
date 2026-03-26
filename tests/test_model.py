import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

MODEL_PATH = "models/sentiment_model.pkl"
DATA_PATH = "data/raw/dataset_etudiants_cameroon.csv"


def test_model_exists():
    """Vérifie que le modèle a bien été généré."""
    assert os.path.exists(MODEL_PATH), "Le modèle n'existe pas."


def test_model_accuracy():
    """Vérifie que l'accuracy du modèle est > 80% comme exigé par le TP."""
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH).dropna(subset=["phrase", "sentiment"])

    # Pour un test CI rapide, on prend un échantillon de 1000 lignes
    df_sample = df.sample(n=1000, random_state=42)
    predictions = model.predict(df_sample["phrase"])

    accuracy = accuracy_score(df_sample["sentiment"], predictions)

    # Raccourcissement de la ligne pour corriger l'erreur E501
    error_msg = f"Accuracy trop faible: {accuracy:.2f} (Attendu > 0.80)"
    assert accuracy > 0.80, error_msg


def test_prediction_format():
    """Vérifie que le modèle renvoie bien une prédiction au bon format."""
    model = joblib.load(MODEL_PATH)
    pred = model.predict(["Les professeurs sont vraiment excellents !"])

    # Raccourcissement de la ligne pour corriger l'erreur E501
    valid_preds = ["positif", "neutre", "négatif"]
    assert pred[0] in valid_preds, "Format de prédiction invalide."
