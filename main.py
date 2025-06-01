# main.py

import os
from data_loader import load_and_preprocess_data
from machine_learning_models import train_and_evaluate_ml_models
from deep_learning_model import train_and_evaluate_dl_model
from evaluation import save_results, plot_training_history

def main():
    directory_path = "D:\\Preparation_for_Github\\Heart Disease prediction"
    file_path = os.path.join(directory_path, 'heart.csv')
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(file_path)
    
    # Train and evaluate machine learning models
    ml_results = train_and_evaluate_ml_models(X_train, X_test, y_train, y_test)
    
    # Train and evaluate deep learning model
    dl_results, history = train_and_evaluate_dl_model(X_train, X_test, y_train, y_test)
    
    # Combine results
    all_results = {**ml_results, "Deep Learning Model": dl_results}
    
    # Save results to a text file
    results_file_path = os.path.join(directory_path, "model_results.txt")
    save_results(all_results, results_file_path)
    
    # Plot training history
    plot_training_history(history, directory_path)

if __name__ == "__main__":
    main()