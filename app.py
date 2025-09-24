import streamlit as st
import joblib
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ---------------------------
# Device setup
# ---------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# Load Transformer model from HF
# ---------------------------
@st.cache_resource
def load_transformer():
    tokenizer = AutoTokenizer.from_pretrained(
        "i-am-turing07/sentiment-analysis-model",
        subfolder="bert_model_zip"
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "i-am-turing07/sentiment-analysis-model",
        subfolder="bert_model_zip"
    )
    model.to(device)
    model.eval()
    return tokenizer, model

# ---------------------------
# Load classical ML models from HF
# ---------------------------
@st.cache_resource
def load_classical_models():
    logistic_path = hf_hub_download("i-am-turing07/sentiment-analysis-model", "logistic_class.pkl")
    nb_path = hf_hub_download("i-am-turing07/sentiment-analysis-model", "naive_bayes.pkl")
    tfidf_path = hf_hub_download("i-am-turing07/sentiment-analysis-model", "tfidf.pkl")

    logistic_model = joblib.load(logistic_path)
    nb_model = joblib.load(nb_path)
    vectorizer = joblib.load(tfidf_path)
    return logistic_model, nb_model, vectorizer

# ---------------------------
# Load models
# ---------------------------
tokenizer, transformer_model = load_transformer()
logistic_model, nb_model, vectorizer = load_classical_models()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("📝 Sentiment Analysis Tool")
st.write(
    "Predict sentiment using Logistic Regression, Naive Bayes, "
    "and Transformer (BERT/DistilBERT), combined with **majority voting**."
)

text = st.text_area("Enter your text here:")

if st.button("Predict Sentiment"):
    if not text.strip():
        st.warning("⚠️ Please enter some text before predicting.")
    else:
        # Classical model predictions
        X = vectorizer.transform([text])
        logistic_pred = logistic_model.predict(X)[0]
        naive_pred = nb_model.predict(X)[0]

        # Transformer prediction
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = transformer_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            transformer_pred = torch.argmax(probs, dim=1).item()

        # Majority Voting
        predictions = [logistic_pred, naive_pred, transformer_pred]
        final = max(set(predictions), key=predictions.count)

        # Sentiment Mapping
        num_labels = transformer_model.config.num_labels
        if num_labels == 2:
            sentiment_map = {0: "Negative", 1: "Positive"}
        elif num_labels == 3:
            sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        else:
            sentiment_map = {i: f"Label {i}" for i in range(num_labels)}

        # Display Results
        st.success(f"Final Sentiment: **{sentiment_map.get(final, final)}**")
        st.write(
            f"- Logistic Regression: {sentiment_map.get(logistic_pred, logistic_pred)}  \n"
            f"- Naive Bayes: {sentiment_map.get(naive_pred, naive_pred)}  \n"
            f"- Transformer: {sentiment_map.get(transformer_pred, transformer_pred)}"
        )
