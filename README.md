# Car Price Prediction

A machine learning project for predicting used car prices from vehicle specifications and characteristics.

## Project Overview

This project follows a complete machine learning workflow, starting with exploratory data analysis and data cleaning, then moving to categorical encoding, model training, hyperparameter tuning, and final evaluation.

The main model used in the final stage is a **Random Forest Regressor**, chosen for its ability to model non-linear relationships between vehicle characteristics and price.

## Dataset

The dataset contains information about used vehicles, including:

- Manufacturer and model
- Production year
- Category
- Engine volume
- Mileage
- Cylinders
- Fuel type
- Gearbox type
- Drive wheels
- Interior and exterior characteristics
- Price

The original dataset contains approximately 19,000 observations.

## Project Workflow

### 1. Exploratory Data Analysis

The EDA stage covers:

- Dataset structure and data types
- Duplicate detection
- Numerical and categorical distributions
- Skewness and kurtosis
- Outlier analysis
- Correlation between numerical variables
- Identification of high-cardinality categorical features

The analysis showed strong skewness and extreme observations in variables such as `Price`, `Levy`, and `Mileage`.

### 2. Data Preprocessing

The preprocessing stage includes:

- Removing duplicate rows
- Converting `Levy`, `Engine volume`, and `Mileage` to numerical values
- Removing extreme outliers using the 3×IQR rule
- Saving the cleaned dataset for the modeling stage

### 3. Feature Preparation and Encoding

Different encoding methods are used depending on the categorical feature:

- **One-Hot Encoding** for categorical variables with a manageable number of categories
- **Target Encoding** for high-cardinality variables such as `Manufacturer` and `Model`
- **Binary encoding** for two-category features such as `Leather interior` and `Wheel`

The encoders are fitted only on the training data and then applied to the test data to avoid data leakage.

### 4. Model Training

The main model is:

**Random Forest Regressor**

The model was evaluated using:

- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of Determination

### 5. Hyperparameter Tuning

RandomizedSearchCV was used to explore the main Random Forest hyperparameters.

A wider search was initially considered, but the combination of many parameter values and cross-validation folds resulted in a large computational cost. The final search was therefore kept smaller, using:

- 10 sampled parameter combinations
- 3-fold cross-validation

This provided a practical compromise between model exploration and computation time.

## Final Model

The final Random Forest configuration used in the project is:

```text
n_estimators = 150
max_depth = 20
min_samples_split = 2
min_samples_leaf = 1
max_features = 0.75
random_state = 42
```

On the held-out test set, the final model achieved approximately:

| Metric | Result |
|---|---:|
| MAE | 4,647 |
| RMSE | 7,862 |
| R² | 0.705 |

The exact values may vary slightly if the preprocessing or model configuration is changed.

## Feature Importance

Random Forest feature importance was also examined to understand which variables contributed most to the model's predictions.

The most influential features include characteristics such as:

- Model
- Production year
- Mileage
- Levy
- Engine volume

Individual one-hot encoded categories generally have smaller importance values because the information from one original categorical variable is distributed across several encoded columns.

## Project Structure

```text
carsML/
│
├── data/
│   ├── car_price_prediction.csv
│   └── car_price_prediction_cleaned.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   └── 03_modeling.ipynb
│
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

Clone the repository and install the required packages:

```bash
pip install -r requirements.txt
```

Then run the notebooks in the following order:

1. `01_eda.ipynb`
2. `02_preprocessing.ipynb`
3. `03_modeling.ipynb`

## Limitations and Possible Improvements

The current model provides a useful baseline, but further improvements could come from:

- More detailed feature engineering
- Testing additional regression algorithms
- Better treatment of rare categorical values
- More systematic feature selection
- Additional model validation
- Further optimization of the hyperparameter search

## Technologies

- Python
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Jupyter Notebook
