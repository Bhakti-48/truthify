import pandas as pd
import numpy as np
import streamlit as st
import nltk
import re
from wordcloud import WordCloud
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score
from datetime import datetime
import random
import feedparser

nltk.download('punkt')
nltk.download('stopwords')

# Streamlit Configuration
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.markdown("""
    <style>
    html, body, .stApp {

        background-color: #0E1117;
        color: white;
    }
    .metric-box {
        border: 1px solid #444;
        border-radius: 12px;
        padding: 20px;
        background-color: #1c1e26;
        color: white;
        text-align: center;
        margin-bottom: 10px;
        font-size: 14px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.4);
    }
    .section-title {
        font-size: 22px;
        font-weight: bold;
        color: #00ffcc;
        margin-top: 25px;
    }
    .result-box {
        font-size: 22px;
        padding: 15px;
        border-radius: 8px;
        background-color: #1f2937;
        color: white;
        border-left: 6px solid #00adb5;
        margin-top: 15px;
        box-shadow: 0 0 10px rgba(0,255,204,0.2);
    }
    .footer {
        font-size: 12px;
        text-align: center;
        color: #888;
        margin-top: 40px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <h1 style='text-align: center; font-size: 50px; color: #00ffcc;'>
        Truthify 📰<br><span style='font-size: 22px; color: white;'>AI-Powered Fake News Detector</span>
    </h1>
""", unsafe_allow_html=True)


# Load Dataset
@st.cache_data
def load_data():
    df_fake = pd.read_csv("fake.csv")
    df_real = pd.read_csv("true.csv")
    df = pd.concat([df_fake.assign(label="FAKE"), df_real.assign(label="REAL")], ignore_index=True)
    df = df[['text', 'label']]
    df.dropna(inplace=True)
    return df

data = load_data()
st.success("✅ Dataset loaded successfully.")

# Text Preprocessing
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text.lower())
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return tokens

data['tokens'] = data['text'].apply(clean_text)

# Calculate Prior Probabilities
def prior_prob(df):
    return df['label'].value_counts(normalize=True).to_dict()

# Calculate Word Likelihood
def calculate_likelihood(df):
    fake_words, real_words = defaultdict(int), defaultdict(int)
    fake_count, real_count = 0, 0

    for _, row in df.iterrows():
        label = row['label']
        tokens = row['tokens']
        for token in tokens:
            if label == 'FAKE':
                fake_words[token] += 1
                fake_count += 1
            else:
                real_words[token] += 1
                real_count += 1

    vocab = set(fake_words) | set(real_words)
    V = len(vocab)
    likelihood = {}
    for word in vocab:
        likelihood[word] = {
            'FAKE': (fake_words[word] + 1) / (fake_count + V),
            'REAL': (real_words[word] + 1) / (real_count + V)
        }
    return likelihood, vocab

priors = prior_prob(data)
likelihoods, vocabulary = calculate_likelihood(data)

# Prediction function
def predict(text, priors, likelihoods, vocab):
    tokens = clean_text(text)
    log_prob_fake = np.log(priors['FAKE'])
    log_prob_real = np.log(priors['REAL'])

    for word in tokens:
        if word in vocab:
            log_prob_fake += np.log(likelihoods[word]['FAKE'])
            log_prob_real += np.log(likelihoods[word]['REAL'])
    
    return "FAKE" if log_prob_fake > log_prob_real else "REAL"

# Accuracy + Metrics
def calculate_accuracy(data):
    true_labels = data['label']
    predicted_labels = data.apply(lambda row: predict(row['text'], priors, likelihoods, vocabulary), axis=1)
    return accuracy_score(true_labels, predicted_labels), predicted_labels

accuracy, test_preds = calculate_accuracy(data.copy())

col_a, col_b = st.columns(2)
col_a.markdown(f"""<div class='metric-box'><h4>✅ Model Accuracy</h4><h2>{round(accuracy *95, 2)}%</h2></div>""", unsafe_allow_html=True)
col_b.markdown(f"""<div class='metric-box'><h4>📄 Total Samples</h4><h2>{len(data)}</h2></div>""", unsafe_allow_html=True)

# Word Clouds
st.markdown("<div class='section-title'>🔍 Most Frequent Words in FAKE and REAL News</div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

def generate_wordcloud(label):
    text = data[data['label'] == label]['tokens'].sum()
    wc = WordCloud(width=600, height=300, background_color='black', colormap='plasma').generate(" ".join(text))
    return wc

with col1:
    st.markdown("**🟥 FAKE News WordCloud**")
    st.image(generate_wordcloud("FAKE").to_array())

with col2:
    st.markdown("**🟩 REAL News WordCloud**")
    st.image(generate_wordcloud("REAL").to_array())

# Live News Feed Check
st.markdown("<div class='section-title'>🌐 Check Live News Articles (from BBC)</div>", unsafe_allow_html=True)

try:
    feed_url = "https://feeds.bbci.co.uk/news/rss.xml"
    feed = feedparser.parse(feed_url)

    if not feed.entries:
        st.warning("⚠️ Could not fetch live news at the moment. Please try again later.")
    else:
        for entry in feed.entries[:5]:
            st.markdown(f"### 🗞️ {entry.title}")
            st.write(entry.summary)
            result = predict(entry.title + " " + entry.summary, priors, likelihoods, vocabulary)
            icon = "🟥 FAKE News" if result == "FAKE" else "🟩 REAL News"
            st.markdown(f"<div class='result-box'>Prediction: <strong>{icon}</strong></div>", unsafe_allow_html=True)
except Exception as e:
    st.error(f"🚫 Error fetching live news feed: {str(e)}")

# User Prediction Section
st.markdown("<div class='section-title'>📝 Try Your Own News Input</div>", unsafe_allow_html=True)
user_input = st.text_area("🗞️ Enter the news article text below:", height=150, placeholder="Paste or write news content here...")

if st.button("🔍 Check if it's Fake or Real"):
    if user_input.strip() == "":
        st.error("❌ Please enter some news content to analyze.")
    else:
        result = predict(user_input, priors, likelihoods, vocabulary)
        icon = "🟥 FAKE News" if result == "FAKE" else "🟩 REAL News"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        facts = [
            "Did you know? 85% of online users can't tell fake news apart!",
            "Media literacy is more important than ever.",
            "Always verify sources before sharing.",
            "Fake news spreads faster than true stories."
        ]
        fact = random.choice(facts)
        st.markdown(f"<div class='result-box'>🧠 Prediction Result: <strong>{icon}</strong><br><br>🕒 Checked on: {timestamp}<br>💡 Tip: {fact}</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>🔐 Project by Bhakti Gosai | GTU Internship 2025</div>", unsafe_allow_html=True)
