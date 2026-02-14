import pandas as pd
import osk

def preprocess_data():
    # Load raw Titanic dataset
    df = pd.read_csv("data/raw/titanic.csv")

    print("Dataset Preview")
    print(df.head())

    print("\nSTARTING PREPROCESSING AND INSPECTION...")

    # Basic dataset inspection
    print("\nDataset Description")
    print(df.describe())

    print("\nDataset Shape")
    print(df.shape)

    print("\nDataset Info")
    print(df.info())

    # -------------------------------
    # Handling Missing Values
    # -------------------------------
    print("\nChecking missing values per column")
    print(df.isnull().sum())

    # Find columns that contain missing values
    null_clos = []
    value = df.isna().sum()

    for i in range(len(value)):
        if value.iloc[i] != 0:
            null_clos.append(df.columns[i])

    print("\nColumns with missing values:", null_clos)

    # Fill missing values
    for col in null_clos:
        if df[col].dtype == "object":
            # Fill categorical columns with mode
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            # Fill numerical columns with median
            df[col] = df[col].fillna(df[col].median())

    print("\nMissing values handled successfully")
    print(df.isna().sum())

    # -------------------------------
    # Encoding Categorical Variables
    # (Only basic encoding, NOT feature engineering)
    # -------------------------------
    print("\nEncoding categorical columns")

    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

    print("Categorical encoding completed")

    # -------------------------------
    # Save processed dataset
    # -------------------------------
    PROCESSED_PATH = "data/processed"
    os.makedirs(PROCESSED_PATH, exist_ok=True)

    processed_file = os.path.join(PROCESSED_PATH, "processed.csv")
    df.to_csv(processed_file, index=False)

    print("\nPreprocessed data saved to:", processed_file)


# Run preprocessing only when the script is executed directly
if __name__ == "__main__":
    preprocess_data()
