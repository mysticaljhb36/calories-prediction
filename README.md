# Calories Expenditure Prediction Using Machine Learning

## Project Overview

This project develops a machine learning solution for predicting calories burned during exercise sessions using physiological and exercise-related measurements.

The project was developed in **Python** using **Jupyter Notebook** and follows a complete end-to-end Data Science workflow including:

* Exploratory Data Analysis (EDA)
* Statistical Analysis
* Normality Testing
* Spearman Rank Correlation Analysis
* Missing Value Treatment using MICE
* Feature Engineering
* Pipeline Development
* Machine Learning Model Comparison
* Hyperparameter Optimisation
* Model Evaluation
* Model Persistence

The objective was to identify the most effective machine learning model for predicting calorie expenditure while applying industry-standard Data Science methodologies.

---

# Key Achievements

* Built an end-to-end Machine Learning pipeline for calorie expenditure prediction.
* Applied **MICE (Multiple Imputation by Chained Equations)** for missing value treatment.
* Engineered nine physiologically meaningful features based on exercise science principles.
* Compared Linear Regression, Random Forest, Extra Trees, and XGBoost models.
* Improved Random Forest performance from **R² = 0.843** to **R² = 0.907** through hyperparameter optimisation.
* Reduced RMSE from **132.04** to **101.52**, significantly reducing large prediction errors.
* Implemented reproducible workflows using Conda environments, setup scripts, and model persistence with Joblib.
* Developed a reusable prediction pipeline for scoring unseen exercise data.

---

# Dataset

## Original Features

| Feature  | Description                        |
| -------- | ---------------------------------- |
| Duration | Exercise duration (minutes)        |
| Pulse    | Average heart rate during exercise |
| Maxpulse | Maximum heart rate achieved        |

## Target Variable

| Variable | Description                     |
| -------- | ------------------------------- |
| Calories | Calories burned during exercise |

---

# Statistical Analysis

## Normality Testing

A normality assessment was performed to determine whether variables followed a normal distribution.

The analysis indicated that the majority of variables were non-parametric (non-normally distributed).

Consequently, **Spearman Rank Correlation** was selected instead of Pearson Correlation because:

* It does not assume normality
* It is robust to skewed distributions
* It is less sensitive to outliers
* It measures monotonic relationships

The resulting correlation coefficients were used to assess feature importance and guide feature engineering decisions.

---

# Missing Value Treatment

Missing values were handled using:

**MICE (Multiple Imputation by Chained Equations)**

implemented using Scikit-Learn's `IterativeImputer`.

MICE was selected because it:

* Preserves relationships between variables
* Reduces bias
* Produces more realistic estimates
* Supports improved machine learning performance

Unlike mean or median imputation, MICE leverages information from all available variables when estimating missing values.

---

# Feature Engineering

Several physiologically meaningful features were created to improve predictive performance.

## Engineered Features

| Feature           | Formula                  |
| ----------------- | ------------------------ |
| Pulse_Reserve     | Maxpulse − Pulse         |
| Pulse_Duration    | Pulse × Duration         |
| Maxpulse_Duration | Maxpulse × Duration      |
| Intensity_Ratio   | Pulse ÷ Maxpulse         |
| Pulse_Squared     | Pulse²                   |
| Duration_Squared  | Duration²                |
| Pulse_Maxpulse    | Pulse × Maxpulse         |
| Log_Duration      | log(1 + Duration)        |
| Heart_Rate_Load   | Duration × (Pulse ÷ 100) |

## Feature Engineering Rationale

These engineered features were designed to capture:

* Cardiovascular reserve
* Exercise intensity
* Cardiovascular workload
* Sustained physiological effort
* Non-linear calorie expenditure relationships

The engineered variables provided additional information beyond the original features and improved model performance.

---

# Machine Learning Pipeline

A machine learning pipeline was developed to ensure reproducibility and minimise the risk of data leakage.

The pipeline incorporated:

1. Data preprocessing
2. MICE imputation
3. Feature engineering
4. Feature scaling (where required)
5. Model training
6. Hyperparameter optimisation
7. Model evaluation

This ensured that identical transformations were applied to both training and unseen datasets.

---

# Models Evaluated

The following machine learning algorithms were compared:

* Linear Regression
* Random Forest Regressor
* Extra Trees Regressor
* XGBoost Regressor

Model performance was evaluated using:

* R² Score
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)

Random Forest Regressor delivered the strongest overall performance and was selected for optimisation.

---

# Baseline Model Performance

A baseline Random Forest Regressor was first trained using default parameters.

| Metric | Baseline RF |
| ------ | ----------: |
| R²     |       0.843 |
| MAE    |       56.12 |
| RMSE   |      132.04 |

The baseline model explained approximately **84.3%** of the variation in calorie expenditure.

---

# Hyperparameter Optimisation

Hyperparameter tuning was performed using **RandomizedSearchCV**.

## Best Parameters

```python
{
    'n_estimators': 200,
    'min_samples_split': 5,
    'min_samples_leaf': 1,
    'max_features': 'sqrt',
    'max_depth': 5,
    'bootstrap': False
}
```

A separate tuned model was then trained using the optimal parameter combination.

---

# Final Model Results

| Metric | Baseline RF | Tuned RF | Improvement |
| ------ | ----------: | -------: | ----------: |
| R²     |       0.843 |    0.907 |      +0.064 |
| MAE    |       56.12 |    53.11 |       -3.02 |
| RMSE   |      132.04 |   101.52 |      -30.52 |

---

# Results Interpretation

## R² (Coefficient of Determination)

The baseline model achieved an R² score of **0.843**, while the tuned model achieved **0.907**.

This indicates:

* Baseline model explains approximately 84.3% of the variation in calorie expenditure.
* Tuned model explains approximately 90.7% of the variation in calorie expenditure.

### R² Interpretation Scale

| R² Score    | Interpretation |
| ----------- | -------------- |
| < 0.50      | Poor           |
| 0.50 – 0.70 | Acceptable     |
| 0.70 – 0.85 | Good           |
| 0.85 – 0.90 | Very Good      |
| > 0.90      | Excellent      |

The final score of **0.907** falls within the **Excellent** category.

### Key Finding

The tuned model can explain approximately **91% of the differences in calorie expenditure between exercise sessions**, demonstrating strong predictive capability from a relatively small set of physiological measurements.

---

## Mean Absolute Error (MAE)

The tuned model achieved an MAE of **53.11 calories**.

This means that predictions are, on average, approximately **53 calories away from the true value**.

For example:

```text
Actual Calories: 500

Typical Prediction Range:
447 – 553 Calories
```

The reduction in MAE was modest, suggesting that optimisation primarily improved overall model fit rather than average prediction accuracy.

---

## Root Mean Squared Error (RMSE)

The tuned model reduced RMSE from:

```text
132.04 → 101.52
```

This indicates:

* Fewer large prediction errors
* Better handling of outlier observations
* Improved model stability
* Stronger generalisation performance

The reduction in RMSE represents the most significant improvement achieved through hyperparameter optimisation.

---

# Project Structure

```text
calories-prediction/
│
├── data/
│   ├── raw/
│   ├── new_data/
│   ├── processed/
│   ├── features/
│   └── result/
│
├── models/
│   ├── power_transformer.pkl
│   └── random_forest_model.pkl
│
├── notebooks/
│   └── calories_prediction.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   ├── predict.py
│   ├── run_pipeline.py
│   ├── paths.py
│   └── utils.py
│
├── requirements.txt
├── environment.yml
├── setup.sh
├── setup.bat
├── README.md
├── LICENSE
└── .gitignore
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/mysticaljhb/calories-prediction.git

cd calories-prediction
```

## Windows Installation (Recommended)

Run:

```cmd
setup.bat
```

Alternatively, double-click:

```text
setup.bat
```

from Windows Explorer.

## Linux / macOS Installation

Make the setup script executable:

```bash
chmod +x setup.sh
```

Run:

```bash
./setup.sh
```

## Manual Conda Installation

Create the environment:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate calories-prediction
```

## Pip Installation

If Conda is unavailable:

```bash
pip install -r requirements.txt
```

---

# Environment Files

| File             | Purpose                                     |
| ---------------- | ------------------------------------------- |
| requirements.txt | Python package dependencies for pip         |
| environment.yml  | Conda environment definition                |
| setup.bat        | Automated environment setup for Windows     |
| setup.sh         | Automated environment setup for Linux/macOS |

---

# Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

or Jupyter Lab:

```bash
jupyter lab
```

Open:

```text
notebooks/calories_prediction.ipynb
```

and execute the notebook cells sequentially.

---

# Predicting New Data

Place new exercise records inside:

```text
data/new_data/
```

Run:

```bash
python src/predict.py
```

Predictions will be generated using:

```text
models/random_forest_model.pkl
```

and any required preprocessing transformations stored within the project.

---

# Reproducing Results

To reproduce the results:

1. Create the environment using one of the installation methods.
2. Launch Jupyter Notebook.
3. Open `calories_prediction.ipynb`.
4. Execute all notebook cells from top to bottom.
5. Review the generated metrics and visualisations.
6. Load the saved model from `models/random_forest_model.pkl` for inference on unseen data.

---

# Technologies Used

* Python
* Pandas
* NumPy
* SciPy
* Scikit-Learn
* XGBoost
* Plotly
* Joblib
* Jupyter Notebook

---

# Future Improvements

Potential future enhancements include:

* Cross-validation optimisation
* Ensemble learning
* SHAP explainability analysis
* FastAPI deployment
* Docker containerisation
* Automated retraining pipelines
* CI/CD integration

---

# Cross-Platform Support

This repository supports:

* Windows (`setup.bat`)
* Linux (`setup.sh`)
* macOS (`setup.sh`)
* Conda environments (`environment.yml`)
* Pip environments (`requirements.txt`)

This ensures the project can be reproduced consistently across multiple operating systems and development environments.

---

# Tags

```text
machine-learning
data-science
random-forest
xgboost
feature-engineering
regression
python
scikit-learn
jupyter-notebook
predictive-analytics
```

---

# Author

**Daniel Okereke**

MSc Data Science | Machine Learning | Predictive Analytics

Portfolio project demonstrating end-to-end machine learning development, statistical analysis, MICE imputation, feature engineering, model optimisation, pipeline construction, model persistence, and reproducible deployment workflows.
