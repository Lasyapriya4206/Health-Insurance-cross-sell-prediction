from sklearn.metrics import accuracy_score, classification_report
import os

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)

    # Create results folder in project root
    os.makedirs("../results", exist_ok=True)

    with open("../results/accuracy.txt", "w") as f:
        f.write(f"Model Accuracy: {acc:.4f}\n\n")
        f.write(report)

    print("Accuracy file saved successfully!")

    return acc, report