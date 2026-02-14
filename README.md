
```markdown
# MLOps Assignment – Automation Using Makefile (Titanic Dataset)

## Overview
This project implements an end-to-end machine learning pipeline using the Titanic dataset.
The complete workflow is automated using GNU Make. No Python script is executed manually.

## Objective
- Build a reproducible ML pipeline
- Automate all steps using a Makefile
- Train a Random Forest classifier
- Generate predictions and evaluation metrics

## Tools & Technologies
- Python
- GNU Make
- pandas
- numpy
- scikit-learn
- joblib

## Project Structure
```

.
├── data/
│   ├── raw/
│   └── processed/
├── features/
├── models/
├── results/
├── src/
│   ├── download_data.py
│   ├── preprocess.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
├── venv/
├── Makefile
├── requirements.txt
└── README.md

````

## Pipeline Steps
1. Dataset download
2. Data preprocessing
3. Feature engineering
4. Model training (Random Forest)
5. Prediction
6. Evaluation

Each step depends on the output of the previous step and is managed by the Makefile.

## Setup Instructions (WSL / Linux)

### 1. Create and activate virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
````

### 2. Install dependencies

```bash
make setup
```

## Running the Pipeline

### Run full pipeline

```bash
make all
```

### Run individual steps

```bash
make download-data
make preprocess
make features
make train
make predict
make evaluate
```

## Outputs

After running `make all`, the following files will be generated:

* `data/raw/titanic.csv`
* `data/processed/processed.csv`
* `features/features.csv`
* `models/model.pkl`
* `results/predictions.csv`
* `results/metrics.txt`

## Cleaning Generated Files

```bash
make clean
```

## Notes

* No Python file should be run directly.
* All execution must be done using Makefile commands.
* The project is reproducible on a clean system using `make all`.

## Author

MLOps Assignment Submission

```

