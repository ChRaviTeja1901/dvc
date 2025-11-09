import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import pickle
import os


def load_vectorized_data(train_path='data/vectorized/train_data_vectorized.csv'):
    train_data = pd.read_csv(train_path)
    return train_data

def evaluation_data(train_data):
    train_data, evaluation_data = train_test_split(train_data, test_size=0.2, random_state=42)
    X_eval = evaluation_data.drop(columns=['label'])
    y_eval = evaluation_data['label']
    return train_data, X_eval, y_eval

def build_model(train_data):
    X_train = train_data.drop(columns=['label'])
    y_train = train_data['label']

    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_eval, y_eval):
    accuracy = model.score(X_eval, y_eval)
    return accuracy

def save_model(model, model_path='models/xgb_model.pkl'):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

train_data = load_vectorized_data()
train_data, X_eval, y_eval = evaluation_data(train_data=train_data)
model = build_model(train_data=train_data)
accuracy = evaluate_model(model, X_eval, y_eval)
print(f"Evaluation Accuracy: {accuracy}")
save_model(model)