import pandas as pd
import string
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)

# ── Load, clean, vectorize, train (same as before) ──────────
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

tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['cleaned'])
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ── Evaluation Metrics ───────────────────────────────────────
print("=" * 45)
print("        MODEL EVALUATION RESULTS")
print("=" * 45)
print(f"Accuracy  : {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Precision : {precision_score(y_test, y_pred) * 100:.2f}%")
print(f"Recall    : {recall_score(y_test, y_pred) * 100:.2f}%")
print(f"F1 Score  : {f1_score(y_test, y_pred) * 100:.2f}%")
print("=" * 45)

print("\nDetailed Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# ── Confusion Matrix ─────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham', 'Spam'],
            yticklabels=['Ham', 'Spam'])
plt.title('Confusion Matrix — Spam Classifier')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(r'C:\TC\confusion_matrix.png')
print("\nConfusion matrix saved to C:\\TC\\confusion_matrix.png")