import os
import pandas as pd
import joblib

def predict():

    # Correct project root path
    CURRENT_FILE_PATH = os.path.abspath(__file__)
    BASE_PATH = os.path.dirname(os.path.dirname(CURRENT_FILE_PATH))

    # Paths
    features_path = os.path.join(BASE_PATH, "features", "features.csv")
    model_path = os.path.join(BASE_PATH, "model", "model.pkl")
    results_dir = os.path.join(BASE_PATH, "results")

    # Load data and model
    df = pd.read_csv(features_path)
    model = joblib.load(model_path)

    print("Starting Prediction")

    # Separate features and target
    X = df.drop(columns=["Survived"], errors="ignore")

    # Generate predictions
    predictions = model.predict(X)

    # Save predictions
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "predictions.csv")

    pd.DataFrame({
        "Prediction": predictions
    }).to_csv(output_path, index=False)

    print("Predictions saved to:", output_path)


if __name__ == "__main__":
    predict()
