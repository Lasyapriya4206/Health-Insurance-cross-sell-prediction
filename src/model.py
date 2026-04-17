import pickle
from sklearn.ensemble import RandomForestClassifier

def train_model(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    with open("models/saved_model.pkl", "wb") as f:
        pickle.dump(model, f)

    return model