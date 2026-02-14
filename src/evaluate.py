import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate():

    # Correct project root path
    CURRENT_FILE_PATH = os.path.abspath(__file__)
    BASE_PATH = os.path.dirname(os.path.dirname(CURRENT_FILE_PATH))

    # Paths
    features_path = os.path.join(BASE_PATH, "features", "features.csv")
    predictions_path = os.path.join(BASE_PATH, "results", "predictions.csv")
    results_dir = os.path.join(BASE_PATH, "results")

    # Load actual data and predictions
    df = pd.read_csv(features_path)
    preds = pd.read_csv(predictions_path)

    print("Starting Evaluation")

    # Actual labels
    y_true = df["Survived"]

    # Predicted labels
    y_pred = preds["Prediction"]

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Save metrics
    os.makedirs(results_dir, exist_ok=True)
    metrics_path = os.path.join(results_dir, "metrics.txt")

    with open(metrics_path, "w") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-Score: {f1:.4f}\n")

    print("Evaluation completed")
    print("Metrics saved to:", metrics_path)


if __name__ == "__main__":
    evaluate()
