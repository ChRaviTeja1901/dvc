import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

def load_data(url, file_path='data.csv'):
    """
    Load data from a URL or local file path.

    Parameters:
    url (str): URL to download the data from.
    file_path (str): Local file path to save/load the data.

    Returns:
    pd.DataFrame: Loaded data as a pandas DataFrame.
    """
    if not os.path.exists(file_path):
        data = pd.read_csv(url)
    else:
        data = pd.read_csv(file_path)
    
    print(f"Data loaded with shape: {data.shape}")
    return data

def encode_data(data):
    """
    Encode categorical variables.

    Parameters:
    data (pd.DataFrame): The raw data.

    Returns:
    pd.DataFrame: Preprocessed data.
    """
    data.drop(columns=['tweet_id'], inplace=True)  # Drop unnecessary columns
    data = data[data['sentiment'].isin(['happiness', 'sadness'])]  # Filter for specific sentiments
    data['sentiment'] = data['sentiment'].replace({'happiness': 1, 'sadness': 0})  # Encode sentiments
    
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    return train_data, test_data

def create_local_copy(train_data, test_data, train_path='train_data.csv', test_path='test_data.csv'):
    """
    Save the preprocessed data to local CSV files.

    Parameters:
    train_data (pd.DataFrame): Training data.
    test_data (pd.DataFrame): Testing data.
    train_path (str): File path to save training data.
    test_path (str): File path to save testing data.
    """
    
    data_path = os.path.join('data', 'raw')
    os.makedirs(data_path, exist_ok=True)
    train_data.to_csv(os.path.join(data_path, train_path), index=False)
    test_data.to_csv(os.path.join(data_path, test_path), index=False)

data = load_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
train_data, test_data = encode_data(data)
create_local_copy(train_data, test_data)