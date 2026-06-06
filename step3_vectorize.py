import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Load and clean dataset (same as before) ─────────────────
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

def clean_text(message):
    message = message.lower()
    message = message.translate(str.maketrans('', '', string.punctuation))
    words = message.split()
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

df['cleaned'] = df['message'].apply(clean_text)

# ── TF-IDF Vectorizer ────────────────────────────────────────
# max_features=3000 means we keep only top 3000 important words
tfidf = TfidfVectorizer(max_features=3000)

# fit_transform learns the vocabulary AND converts text to numbers
X = tfidf.fit_transform(df['cleaned'])

# ── Labels ───────────────────────────────────────────────────
# Convert spam/ham to 1/0 (numbers) for the model
df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})
y = df['label_num']

# ── Show results ─────────────────────────────────────────────
print("Shape of feature matrix X:", X.shape)
print("Total messages:", X.shape[0])
print("Total features (words):", X.shape[1])

print("\nLabel encoding:")
print(df[['label', 'label_num']].head(10))

print("\nTop 20 words in vocabulary:")
print(tfidf.get_feature_names_out()[:20])