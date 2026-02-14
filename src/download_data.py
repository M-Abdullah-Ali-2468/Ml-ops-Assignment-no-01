import pandas as pd
import os

# This function downloads the Titanic dataset
# and saves it inside the data/raw directory
def download_data():

    # URL of the Titanic dataset
    DATASET_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

    # Name of the file to save locally
    DATASET_NAME = "titanic.csv"

    # Base path of the project (Makefile is run from project root)
    CURRENT_FILE_PATH = os.path.abspath(__file__)
    BASE_PATH = os.path.dirname(os.path.dirname(CURRENT_FILE_PATH))


    # Path where raw data will be stored
    RAW_DATA_PATH = os.path.join(BASE_PATH, "data/raw")

    # Create the raw data directory if it does not exist
    os.makedirs(RAW_DATA_PATH, exist_ok=True)

    # Full path of the dataset file
    DATASET_PATH = os.path.join(RAW_DATA_PATH, DATASET_NAME)

    # Download the dataset only if it does not already exist
    if not os.path.exists(DATASET_PATH):
        print("Downloading dataset...")
        df = pd.read_csv(DATASET_URL)
        df.to_csv(DATASET_PATH, index=False)

    # Confirmation message
    print("Dataset downloaded successfully!")

# This block ensures the script runs only
# when executed directly (not when imported)
if __name__ == "__main__":
    download_data()
