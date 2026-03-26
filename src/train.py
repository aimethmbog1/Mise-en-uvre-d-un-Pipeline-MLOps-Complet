import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

DATA_PATH = "data/raw/dataset_etudiants_cameroon.csv"
MODEL_PATH = "models/sentiment_model.pkl"


def train():
    print("⏳ Chargement des données...")
    df = pd.read_csv(DATA_PATH)

    # Nettoyage rapide et préparation
    df = df.dropna(subset=["phrase", "sentiment"])
    X = df["phrase"]
    y = df["sentiment"]  # 'positif', 'neutre', 'négatif'

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("🧠 Entraînement du modèle (TF-IDF + Logistic Regression)...")
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    pipeline.fit(X_train, y_train)

    # Évaluation
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy sur le jeu de test : {acc:.2f}")

    # Sauvegarde
    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"💾 Modèle sauvegardé dans {MODEL_PATH}")


if __name__ == "__main__":
    train()
