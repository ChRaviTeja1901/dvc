import numpy as np
import pandas as pd
import os
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer

nltk.download('stopwords')
nltk.download('wordnet')


# load the data from data/raw
def load_data(train_path='data/raw/train_data.csv', test_path='data/raw/test_data.csv'):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data


# preprocess the text data
def lowercase_text(text):
    return str(text).lower()

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_punctuation(text):
    text = re.sub(r'[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_numbers(text):
    return re.sub(r'\d+', '', str(text))

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

def fillna_with_empty(text):
    if pd.isna(text):
        return ''
    return text

def preprocess_data(train_data, test_data):
    for dataset in [train_data, test_data]:
        dataset['content'] = dataset['content'].apply(fillna_with_empty)
        dataset['content'] = dataset['content'].apply(lowercase_text)
        dataset['content'] = dataset['content'].apply(remove_urls)
        dataset['content'] = dataset['content'].apply(remove_punctuation)
        dataset['content'] = dataset['content'].apply(remove_numbers)
        dataset['content'] = dataset['content'].apply(remove_stopwords)
        dataset['content'] = dataset['content'].apply(lemmatize_text)

    return train_data, test_data

# save the preprocessed data to data/processed
def save_preprocessed_data(train_data, test_data, train_path='train_data_processed.csv', test_path='test_data_processed.csv'):
    data_path = os.path.join('data', 'processed')
    os.makedirs(data_path, exist_ok=True)
    train_data.to_csv(os.path.join(data_path, train_path), index=False)
    test_data.to_csv(os.path.join(data_path, test_path), index=False)


train_data, test_data = load_data()
train_data, test_data = preprocess_data(train_data, test_data)
save_preprocessed_data(train_data, test_data)