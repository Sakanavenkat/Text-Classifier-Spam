import pandas as pd
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# ── Load and clean dataset ───────────────────────────────────
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
df['label_num'] = df['label'].map({'spam': 1, 'ham': 0})

# ── TF-IDF Vectorization ─────────────────────────────────────
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['cleaned'])
y = df['label_num']

# ── Train/Test Split ─────────────────────────────────────────
# test_size=0.2 means 20% for testing, 80% for training
# random_state=42 means results are same every time you run
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training messages:", X_train.shape[0])
print("Testing messages:", X_test.shape[0])

# ── Train Naive Bayes Model ──────────────────────────────────
model = MultinomialNB()

# fit() is where the model actually LEARNS from training data
model.fit(X_train, y_train)
print("\nModel training complete!")

# ── Test the model ───────────────────────────────────────────
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# ── Test with your own message ───────────────────────────────
def predict_message(message):
    cleaned = clean_text(message)
    vectorized = tfidf.transform([cleaned])
    prediction = model.predict(vectorized)
    return "SPAM 🚨" if prediction[0] == 1 else "HAM ✅"

print("\n--- Test your own messages ---")
print(predict_message("Congratulations! You won a free prize click now"))
print(predict_message("Hey are you coming to class tomorrow?"))
print(predict_message("FREE entry WIN cash prize call now!"))
print(predict_message("Can we meet for lunch today?"))