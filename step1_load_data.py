import pandas as pd

# Load dataset directly from GitHub (always works!)
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])

print("Shape of dataset:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nLabel distribution:")
print(df['label'].value_counts())
print("\nOne example SPAM message:")
print(df[df['label'] == 'spam']['message'].iloc[0])
print("\nOne example HAM message:")
print(df[df['label'] == 'ham']['message'].iloc[0])