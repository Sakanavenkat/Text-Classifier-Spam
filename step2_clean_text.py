import pandas as pd
import nltk
import string
from nltk.corpus import stopwords

# ── Load dataset (same as Step 1) ──────────────────────────
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

# ── Cleaning function ───────────────────────────────────────
def clean_text(message):

    # 1. Lowercase — "FREE" becomes "free"
    message = message.lower()

    # 2. Remove punctuation — "hello!" becomes "hello"
    message = message.translate(str.maketrans('', '', string.punctuation))

    # 3. Tokenize — "go home now" becomes ["go", "home", "now"]
    words = message.split()

    # 4. Remove stopwords — removes "the", "is", "a" etc.
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]

    # 5. Join back — ["go", "home", "now"] becomes "go home now"
    return ' '.join(words)

# ── Apply cleaning to every message ────────────────────────
df['cleaned'] = df['message'].apply(clean_text)

# ── Show results ────────────────────────────────────────────
print("BEFORE cleaning:")
print(df['message'].iloc[2])

print("\nAFTER cleaning:")
print(df['cleaned'].iloc[2])

print("\nFirst 5 cleaned messages:")
print(df[['message', 'cleaned']].head())