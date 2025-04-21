from flask import Flask, render_template, request
import pickle
import re
import nltk
import os
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# Download stopwords (only once)
nltk.download('stopwords', quiet=True)

# Initialize Flask app
app = Flask(__name__)

# Load stopwords set
stopwords_set = set(stopwords.words('english'))

# Emoticon pattern
emoticon_pattern = re.compile(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)')

# Define file paths for model and vectorizer
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
model_path = os.path.join(BASE_DIR, 'clf.pkl')
vectorizer_path = os.path.join(BASE_DIR, 'tfidf.pkl')

# Load trained classifier and vectorizer
clf = None
tfidf = None

if os.path.exists(model_path) and os.path.exists(vectorizer_path):
    with open(model_path, 'rb') as f:
        clf = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        tfidf = pickle.load(f)
else:
    print("[ERROR] Model or vectorizer file not found.")

# Text preprocessing function
def preprocessing(text):
    text = re.sub('<[^>]*>', '', text)  # Remove HTML tags
    emojis = emoticon_pattern.findall(text)
    text = re.sub(r'\W+', ' ', text.lower()) + ' '.join(emojis).replace('-', '')
    porter = PorterStemmer()
    words = [porter.stem(word) for word in text.split() if word not in stopwords_set]
    return " ".join(words)

# Home route for sentiment analysis
@app.route('/', methods=['GET', 'POST'])
def analyze_sentiment():
    sentiment = None
    comment = ""

    if request.method == 'POST':
        comment = request.form.get('comment', '')

        if clf is None or tfidf is None:
            sentiment = "Model not available. Please check your files."
        else:
            preprocessed = preprocessing(comment)
            vector = tfidf.transform([preprocessed])
            sentiment = clf.predict(vector)[0]

        return render_template('index.html', sentiment=sentiment, comment=comment)

    return render_template('index.html', comment=comment)

# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)
