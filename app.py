import gradio as gr
import joblib

# Chargement du modèle
try:
    model = joblib.load("models/sentiment_model.pkl")
except Exception:  # Correction de l'erreur E722
    model = None


def predict_sentiment(text):
    if model is None:
        return "Erreur: Modèle non trouvé. Veuillez exécuter train.py"
    prediction = model.predict([text])[0]
    emojis = {"positif": "🟢", "neutre": "🟡", "négatif": "🔴"}
    return f"{prediction.capitalize()} {emojis.get(prediction, '')}"


# Création de l'interface Gradio
interface = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Entrez un avis sur l'université..."),
    outputs=gr.Textbox(label="Sentiment Prédit"),
    title="🎓 Analyse de Sentiments - Avis Étudiants",
    # Les chaînes de caractères longues sont coupées pour éviter l'erreur E501
    description=(
        "Application MLOps pour prédire la polarité " "d'un avis universitaire."
    ),
)

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860)
