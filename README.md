 🎬 YouTube Comment Sentiment Analysis
📌 Project Overview

This project analyzes YouTube trailer comments to understand audience sentiment and predict how the movie might perform at the box office.
By applying NLP techniques and state-of-the-art transformer models, we classify comments as positive or negative, calculate sentiment scores, and provide a consolidated sentiment report.
🚀 Features
* Fetch or load YouTube comments from a file (CSV/Excel).
* Preprocess text (tokenization, stopword removal).
* Perform sentiment analysis using:
  * VADER (Valence Aware Dictionary and sEntiment Reasoner)
  * HuggingFace Transformers (DistilBERT)
* Generate sentiment distribution reports.
* Create positive and negative word clouds for visualization.
🛠️ Tech Stack

* Python
* PyTorch
* HuggingFace Transformers
* NLTK (Natural Language Toolkit)
* VADER Sentiment Analysis

 📂 Project Structure

```
final_project_sentiment_analysics_from_comment.ipynb   # Main Colab Notebook
data/                                                 # Input dataset (YouTube comments in CSV/Excel)
outputs/                                              # Sentiment results & word clouds
```

## ⚙️ Installation

Run the following commands inside Google Colab or your local environment:

```bash
!pip install torch
!pip install transformers
!pip install nltk
!pip install vaderSentiment
```

Download required NLTK data:

```python
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt_tab')
```

## ▶️ Usage

1. Upload your YouTube comments dataset (CSV/Excel) into the notebook.
2. Run preprocessing functions (stopword removal, tokenization).
3. Perform sentiment analysis using VADER & Transformers.
4. Generate sentiment score report + word clouds.

Example:

```python
# Sentiment analysis example
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
classifier("I absolutely loved this trailer!")
# Output: [{'label': 'POSITIVE', 'score': 0.999}]
```
 📊 Results

* **Overall sentiment distribution** (positive vs negative).
* **Word clouds** highlighting the most frequent positive and negative words.
* Final sentiment report for predicting audience reception.

## 📈 Future Work

* Automate fetching comments via **YouTube API**.
* Improve accuracy with fine-tuned transformer models.
* Deploy as a web app for real-time comment analysis.

## 🤝 Contributing

Contributions are welcome! Please fork this repository and submit a pull request.
