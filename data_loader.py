# data_loader.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path):
    # Loading the dataset
    df = pd.read_csv(file_path)
    
    # Converting the categorical variables into dummy variables using 'get_dummies()'
    dataset = pd.get_dummies(df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])
    
    # Feature Scaling
    standScaler = StandardScaler()
    columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    dataset[columns_to_scale] = standScaler.fit_transform(dataset[columns_to_scale])
    
    # Splitting the dataset into dependent and independent features
    X = dataset.drop('target', axis=1)
    y = dataset['target']
    
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test