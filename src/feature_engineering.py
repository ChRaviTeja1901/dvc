import numpy as np
import pandas as pd
import os
import yaml

from sklearn.feature_extraction.text import CountVectorizer

# load the processed data from data/processed
def load_processed_data(train_path='data/processed/train_data_processed.csv', test_path='data/processed/test_data_processed.csv'):
    """
    Load processed data from CSV files.
    """
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

def load_params(param_path='params.yaml'):
    """
    Load parameters from a YAML file.
    """
    with open(param_path, 'r') as f:
        params = yaml.safe_load(f)
        max_features = params['feature_engineering']['max_features']
    return max_features

# vectorize the text data
def vectorize_text(train_data, test_data, max_features=None):
    """
    Vectorize the text data using CountVectorizer.
    """
    vectorizer = CountVectorizer(max_features=max_features)
    train_data['content'] = train_data['content'].fillna('').astype(str)
    test_data['content'] = test_data['content'].fillna('').astype(str)

    X_train = vectorizer.fit_transform(train_data['content'])
    X_test = vectorizer.transform(test_data['content'])

    train_df = pd.DataFrame(X_train.toarray(), columns=vectorizer.get_feature_names_out())
    test_df = pd.DataFrame(X_test.toarray(), columns=vectorizer.get_feature_names_out())

    train_df['label'] = train_data['sentiment'].values
    test_df['label'] = test_data['sentiment'].values

    return train_df, test_df

# save the vectorized data to data/vectorized
def save_vectorized_data(train_data, test_data, train_path='train_data_vectorized.csv', test_path='test_data_vectorized.csv'):
    """
    Save the vectorized data to CSV files.
    """
    data_path = os.path.join('data', 'vectorized')
    os.makedirs(data_path, exist_ok=True)
    train_data.to_csv(os.path.join(data_path, train_path), index=False)
    test_data.to_csv(os.path.join(data_path, test_path), index=False)

def main():
    """Main function to execute feature engineering."""
    train_data, test_data = load_processed_data()
    max_features = load_params()
    vectorized_train_data, vectorized_test_data = vectorize_text(train_data, test_data, max_features)
    save_vectorized_data(vectorized_train_data, vectorized_test_data)


if __name__ == '__main__':
    main()