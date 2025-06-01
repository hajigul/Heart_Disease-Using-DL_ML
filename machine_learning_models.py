# machine_learning_models.py

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score

def train_and_evaluate_ml_models(X_train, X_test, y_train, y_test):
    models = {
        "KNeighbors Classifier": KNeighborsClassifier(n_neighbors=12),
        "Decision Tree Classifier": DecisionTreeClassifier(max_depth=3),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=90),
        "SVM Classifier": SVC(),
        "Logistic Regression": LogisticRegression(max_iter=1000)
    }
    
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        results[name] = {"accuracy": accuracy, "precision": precision, "recall": recall}
    
    return results