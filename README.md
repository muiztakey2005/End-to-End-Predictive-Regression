# End-to-End Predictive Regression Pipeline with Model Metrics

An end-to-end machine learning project for predicting residential property sale prices using an Ames Housing-style dataset. The project covers exploratory data analysis, data cleaning, preprocessing, multiple regression models, cross-validation, hyperparameter tuning, final evaluation, residual analysis, and model persistence.

---

## Project Overview

Accurately predicting house prices requires handling both numerical and categorical property characteristics. This project develops and evaluates multiple regression models to predict `SalePrice`.

The complete workflow includes:

* Exploratory Data Analysis (EDA)
* Missing-value analysis and data cleaning
* Feature and target separation
* Numerical and categorical preprocessing
* Multiple regression models
* Model evaluation using MAE, MSE, RMSE, and R²
* 5-fold cross-validation
* Hyperparameter tuning using GridSearchCV
* Final evaluation on an untouched test set
* Actual vs. predicted analysis
* Residual analysis
* Prediction error analysis
* Model persistence using Joblib

---

## Dataset

The project uses an Ames Housing-style dataset containing **2,051 observations and 81 columns**.

### Target Variable

`SalePrice`

### Features

The dataset contains numerical and categorical property characteristics related to:

* Overall property quality
* Living area
* Basement characteristics
* Garage characteristics
* Lot characteristics
* Property location
* Construction and building features
* Sale-related information

The columns `Id` and `PID` were excluded from model features because they are identifiers rather than meaningful predictive variables.

---

## Exploratory Data Analysis

The EDA phase investigated:

* Dataset dimensions and structure
* Data types
* Missing values
* SalePrice distribution
* Feature correlations
* Important numerical relationships
* Neighborhood-level price distributions
* SalePrice outliers
* Overall quality versus SalePrice

The target variable showed a right-skewed distribution with a long upper tail.

Strong relationships with `SalePrice` were observed for important property characteristics such as:

* `Overall Qual`
* `Gr Liv Area`
* `Garage Area`
* `Garage Cars`
* `Total Bsmt SF`
* `1st Flr SF`

High SalePrice observations were retained when they represented legitimate properties rather than being removed solely because they were statistical outliers.

---

## Data Preprocessing

The preprocessing workflow handles numerical and categorical variables separately.

### Numerical Features

1. Missing values are imputed using the median.
2. Features are standardized using `StandardScaler`.

### Categorical Features

1. Missing values are imputed using the most frequent category.
2. Categories are converted into numerical representations using `OneHotEncoder`.
3. `handle_unknown='ignore'` is used to safely process unseen categories.

A `ColumnTransformer` combines both preprocessing pipelines.

### Feature Distribution

After feature selection:

* Numerical features: **36**
* Categorical features: **42**
* Total model input features before encoding: **78**

---

## Train-Test Split

The dataset was divided into:

* Training set: **1,640 observations**
* Test set: **411 observations**

The split used:

```text
test_size = 0.20
random_state = 42
```

The test set was kept separate for final model evaluation.

---

## Models Evaluated

Four baseline regression approaches were evaluated:

1. Linear Regression
2. Ridge Regression
3. Random Forest Regressor
4. Gradient Boosting Regressor

Gradient Boosting was subsequently tuned using `GridSearchCV`.

---

## Evaluation Metrics

The following regression metrics were used:

### Mean Absolute Error (MAE)

Measures the average absolute difference between actual and predicted values.

### Mean Squared Error (MSE)

Measures the average squared prediction error.

### Root Mean Squared Error (RMSE)

The square root of MSE, expressed in the same units as the target variable.

### R² Score

Measures the proportion of variance in the target explained by the model.

For this project, lower MAE/RMSE indicates smaller prediction errors, while a higher R² indicates greater explained variance.

---

## Model Comparison

| Model                       |           MAE |                MSE |          RMSE |         R² |
| --------------------------- | ------------: | -----------------: | ------------: | ---------: |
| **Tuned Gradient Boosting** | **13,539.34** | **382,556,393.40** | **19,559.05** | **0.9356** |
| Gradient Boosting           |     13,955.26 |     409,422,806.85 |     20,234.20 |     0.9311 |
| Linear Regression           |     14,709.34 |     443,287,470.53 |     21,054.39 |     0.9254 |
| Ridge Regression            |     15,402.15 |     506,028,172.84 |     22,495.07 |     0.9148 |
| Random Forest               |     15,099.46 |     531,930,711.69 |     23,063.62 |     0.9105 |

The tuned Gradient Boosting configuration produced the lowest RMSE among the tested configurations and achieved an R² of **0.9356** on the held-out test set.

---

## Hyperparameter Tuning

`GridSearchCV` with 5-fold cross-validation was used to tune the Gradient Boosting model.

### Parameters Evaluated

```text
n_estimators:
200, 300, 400

learning_rate:
0.03, 0.05, 0.08

max_depth:
2, 3, 4
```

This resulted in:

```text
27 parameter combinations
5-fold cross-validation
135 total fits
```

### Best Parameters

```text
n_estimators = 400
learning_rate = 0.08
max_depth = 3
```

### Best Cross-Validation RMSE

```text
22,797.08
```

The final tuned model was then evaluated on the untouched test set.

---

## Final Model Performance

The tuned Gradient Boosting model achieved:

```text
MAE  : 13,539.34
MSE  : 382,556,393.40
RMSE : 19,559.05
R²   : 0.9356
```

Compared with the untuned Gradient Boosting model:

| Metric | Before Tuning |  After Tuning |
| ------ | ------------: | ------------: |
| MAE    |     13,955.26 | **13,539.34** |
| RMSE   |     20,234.20 | **19,559.05** |
| R²     |        0.9311 |    **0.9356** |

Hyperparameter tuning therefore improved the tested holdout performance of the Gradient Boosting model.

---

## Model Diagnostics

### Actual vs Predicted

The actual-versus-predicted plot showed that many predictions were concentrated around the perfect-prediction reference line. Some larger deviations were observed, particularly among higher-priced properties.

### Residual Analysis

Residuals were generally distributed around zero, indicating no strong systematic bias. However, greater residual dispersion was observed for some higher-priced properties, suggesting increased prediction uncertainty in that range.

### Error Analysis

The 10 largest absolute prediction errors were examined individually. Several properties showed substantial differences between their actual and predicted sale prices.

These observations were retained rather than removed solely because they produced large prediction errors.

---

## Cross-Validation

Five-fold cross-validation was performed on the training data for the Gradient Boosting model.

Results:

```text
Mean MAE  : 14,670.40
Std MAE   : 904.24

Mean RMSE : 23,257.95
Std RMSE  : 1,430.22

Mean R²   : approximately 0.91
Std R²    : approximately 0.02
```

The cross-validation results provide an estimate of performance variability across different training/validation splits.

---

## Model Persistence

The complete tuned preprocessing and regression pipeline was saved using Joblib:

```text
models/
└── tuned_gradient_boosting_pipeline.pkl
```

The saved pipeline was subsequently loaded and tested. Predictions from the loaded model matched the original model exactly, with a maximum prediction difference of:

```text
0.0
```

This verifies that the persisted pipeline can reproduce the original predictions.

---

## Project Structure

```text
End-to-End-Predictive-Regression/
│
├── notebook/
│   └── predictive_regression_pipeline.ipynb
│
├── models/
│   └── tuned_gradient_boosting_pipeline.pkl
│
├── results/
│   ├── final_model_metrics.csv
│   └── model_comparison.csv
│
├── README.md
└── requirements.txt
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Google Colab
* Git & GitHub

---

## How to Use

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd End-to-End-Predictive-Regression
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Open the notebook

Open:

```text
notebook/predictive_regression_pipeline.ipynb
```

### 4. Load the trained model

```python
import joblib

model = joblib.load(
    'models/tuned_gradient_boosting_pipeline.pkl'
)
```

The saved pipeline contains both preprocessing and the trained regression model.

---

## Limitations

* Evaluation is based on a single held-out test set.
* Cross-validation and test performance can vary with different data splits.
* Prediction errors are larger for some higher-priced properties.
* The model has been evaluated on the available Ames Housing-style dataset and has not been externally validated on a different housing market.
* Model performance should therefore not be interpreted as guaranteed performance on new geographic markets or future housing data.

---

## Future Improvements

Potential extensions include:

* Testing additional boosting algorithms such as XGBoost or LightGBM
* Feature engineering based on domain knowledge
* Target transformation to address SalePrice skewness
* More extensive hyperparameter optimization
* Feature importance and model interpretability analysis
* External validation on another housing dataset
* Deployment through a Flask or FastAPI prediction API
* Interactive prediction interface

---

## Author

**Muiz Takey**

Final-Year BE Computer Engineering
Specialization: Artificial Intelligence & Machine Learning

---

## Project Objective

The objective of this project is to demonstrate a complete and reproducible machine learning regression workflow—from exploratory analysis and preprocessing through model comparison, cross-validation, hyperparameter tuning, final evaluation, error analysis, and model deployment preparation.
