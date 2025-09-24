# 📝 Sentiment Analysis Tool

This is a **Streamlit-based Sentiment Analysis Tool** that predicts sentiment using **classical machine learning models** and a **transformer-based model**. The final sentiment is determined using **majority voting**.  

The models used here (Logistic Regression, Naive Bayes, and DistilBERT) are hosted on [Hugging Face](https://huggingface.co/i-am-turing07/sentiment-analysis-model).  

If you want to see the code for **generating these models**, check out the research repository: [Sentiment Analysis Research Project](https://github.com/raman976/sentiment_analysis).

---

## Features

- Predicts sentiment using:
  - **Logistic Regression** (classical)
  - **Naive Bayes** (classical)
  - **DistilBERT** (transformer-based)
- **Majority voting** to combine predictions
- Interactive and user-friendly **Streamlit interface**
- Supports single text input for quick predictions

---

## How It Works

1. The input text is preprocessed.
2. Classical ML models (Logistic Regression and Naive Bayes) generate predictions.
3. DistilBERT generates transformer-based predictions.
4. Majority voting decides the final sentiment: **Negative**, **Neutral**, or **Positive**.
5. The results are displayed on the Streamlit app.

---


## Installation

```bash
git clone https://github.com/raman976/sentiment-analysis-tool.git
cd sentiment-analysis-tool
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install --upgrade pip
pip install -r requirements.txt
