import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

def feature_engineering():

    # Project root directory
    BASE_PATH = os.getcwd()

    # Correct absolute path for input file
    input_path = os.path.join(BASE_PATH, "data", "processed", "processed.csv")

    # Load preprocessed dataset
    df = pd.read_csv(input_path)

    print("Starting Feature Engineering")

    # ---------------------------------
    # Remove unnecessary columns
    # ---------------------------------
    drop_columns = ["PassengerId", "Name", "Ticket", "Cabin"]
    df = df.drop(columns=[col for col in drop_columns if col in df.columns])

    # ---------------------------------
    # Create FamilySize feature
    # ---------------------------------
    if "SibSp" in df.columns and "Parch" in df.columns:
        df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
        df = df.drop(columns=["SibSp", "Parch"])

    # Extra feature
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # ---------------------------------
    # Scaling (exclude target)
    # ---------------------------------
    target_col = "Survived"
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # ---------------------------------
    # Save features
    # ---------------------------------
    output_dir = os.path.join(BASE_PATH, "features")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "features.csv")
    df.to_csv(output_path, index=False)

    print("Feature engineering completed")
    print("Saved to:", output_path)


if __name__ == "__main__":
    feature_engineering()
