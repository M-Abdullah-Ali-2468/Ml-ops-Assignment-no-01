import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model():

    # Get project root path
    CURRENT_FILE_PATH = os.path.abspath(__file__)
    BASE_PATH = os.path.dirname(os.path.dirname(CURRENT_FILE_PATH))

    # Load engineered features
    features_path = os.path.join(BASE_PATH, "features", "features.csv")
    df = pd.read_csv(features_path)

    print("Starting Model Training")

    # Separate features and target
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Train model
    model.fit(X_train, y_train)
    print("Model training completed")

    # Save trained model
    model_dir = os.path.join(BASE_PATH, "model")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(model, model_path)

    print("Model saved at:", model_path)


if __name__ == "__main__":
    train_model()
