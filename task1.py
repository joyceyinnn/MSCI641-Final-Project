import json
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from google.colab import files
uploaded = files.upload()

# Define a function to load jsonl data into a pandas DataFrame
def load_jsonl_to_dataframe(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# Load train, validation, and test data
train_df = load_jsonl_to_dataframe('train.jsonl')
val_df = load_jsonl_to_dataframe('val.jsonl')
test_df = load_jsonl_to_dataframe('test.jsonl')

# Display the first few rows of the train DataFrame
train_df.head()

# check missing value
missing_values = train_df.isnull().sum()
missing_values

# fill missing value
train_df['targetDescription'].fillna('', inplace=True)
train_df['targetKeywords'].fillna('', inplace=True)
train_df['targetMedia'].fillna('', inplace=True)
train_df['targetUrl'].fillna('', inplace=True)

# drop unnecessary columns
train_df.drop(columns=['targetMedia', 'targetUrl'], inplace=True)

# Compute and add new feature columns
train_df['postTextLength'] = train_df['postText'].apply(lambda x: len(' '.join(x)))
train_df['targetParagraphCount'] = train_df['targetParagraphs'].apply(len)
train_df['spoilerLength'] = train_df['spoiler'].apply(lambda x: len(' '.join(x)))

train_df.head()

#transfer to single string
train_df['tags_str'] = train_df['tags'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
train_df[['tags', 'tags_str']].head()

# labeling
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
train_df['encodedTags'] = label_encoder.fit_transform(train_df['tags_str'])

unique_tags_encoded = train_df[['tags_str', 'encodedTags']].drop_duplicates()
unique_tags_encoded


# Combine 'postText' and 'targetParagraphs' into single text features for TF-IDF vectorization
train_df['text'] = train_df['postText'].apply(' '.join) + ' ' + train_df['targetParagraphs'].apply(' '.join)
val_df['text'] = val_df['postText'].apply(' '.join) + ' ' + val_df['targetParagraphs'].apply(' '.join)

# Extract labels
train_df['tags_str'] = train_df['tags'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else '')
val_df['tags_str'] = val_df['tags'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else '')

# Encode labels
label_encoder = LabelEncoder()
train_df['encodedTags'] = label_encoder.fit_transform(train_df['tags_str'])
val_df['encodedTags'] = label_encoder.transform(val_df['tags_str'])

# Calculate additional features
train_df['postTextLength'] = train_df['postText'].apply(lambda x: len(' '.join(x)))
train_df['targetParagraphCount'] = train_df['targetParagraphs'].apply(len)
val_df['postTextLength'] = val_df['postText'].apply(lambda x: len(' '.join(x)))
val_df['targetParagraphCount'] = val_df['targetParagraphs'].apply(len)

# Define additional feature columns
additional_features = ['postTextLength', 'targetParagraphCount']

# Define TF-IDF vectorizer with n-grams
tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

# Define a custom transformer to select additional features
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.feature_names]

# Combine TF-IDF and additional features using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('tfidf', tfidf_vectorizer, 'text'),
        ('additional_features', StandardScaler(), additional_features)
    ])

# Define the model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=200, random_state=42))
])

# Prepare feature and target variables
X_train = train_df[['text', 'postTextLength', 'targetParagraphCount']]
X_val = val_df[['text', 'postTextLength', 'targetParagraphCount']]
y_train = train_df['encodedTags']
y_val = val_df['encodedTags']

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_val)

# Classified output
classification_report_val = classification_report(y_val, y_pred, target_names=label_encoder.classes_)
print(classification_report_val)

# Load test data
test_df = pd.read_json('test.jsonl', lines=True)

# Combine 'postText' and 'targetParagraphs' into single text features for TF-IDF vectorization
test_df['text'] = test_df['postText'].apply(' '.join) + ' ' + test_df['targetParagraphs'].apply(' '.join)

# Calculate additional features for test data
test_df['postTextLength'] = test_df['postText'].apply(lambda x: len(' '.join(x)))
test_df['targetParagraphCount'] = test_df['targetParagraphs'].apply(len)

# Prepare test features
X_test = test_df[['text', 'postTextLength', 'targetParagraphCount']]

# Predict tags for the test set
test_df['predictedTags'] = label_encoder.inverse_transform(model.predict(X_test))

# Create submission DataFrame
submission_df = pd.DataFrame({
    'id': test_df['postId'],
    'spoilerType': test_df['predictedTags']
})

# Generate CSV file
submission_df.to_csv('prediction_task1.csv', index=False)
print("Predictions saved to 'prediction_task1.csv'")

#download csv file
files.download("prediction_task1.csv")