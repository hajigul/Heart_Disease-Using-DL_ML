# deep_learning_model.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score

def train_and_evaluate_dl_model(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)
    
    y_pred_dl = (model.predict(X_test) > 0.5).astype("int32")
    accuracy_dl = accuracy_score(y_test, y_pred_dl)
    precision_dl = precision_score(y_test, y_pred_dl)
    recall_dl = recall_score(y_test, y_pred_dl)
    
    return {"accuracy": accuracy_dl, "precision": precision_dl, "recall": recall_dl}, history