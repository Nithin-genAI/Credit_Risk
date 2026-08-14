# Credit Risk Modeling Project

## Project Overview
This project aims to build a reliable machine learning model for predicting credit risk using the Home Credit Default Risk dataset. The goal is to predict whether a loan applicant will default on their payment (TARGET = 1) or not (TARGET = 0).

## Dataset Information
- **Source**: Home Credit Default Risk dataset from Kaggle
- **Size**: 307,511 entries with 122 features
- **Target Variable**: TARGET (0 = non-defaulter, 1 = defaulter)
- **Default Rate**: ~8.07% (highly imbalanced dataset)

## Key Features Explored
1. **Numerical Features**: 106 features including income, credit amounts, annuity payments, etc.
2. **Categorical Features**: 16 features including contract type, gender, education, housing type, etc.
3. **External Sources**: EXT_SOURCE_1, EXT_SOURCE_2, EXT_SOURCE_3 (external data sources)
4. **Time-based Features**: DAYS_BIRTH, DAYS_EMPLOYED, etc.
5. **Aggregated Features**: Various average, mode, median calculations for living areas, etc.

## Data Preprocessing Steps Completed
1. **Data Loading**: Successfully loaded the dataset (5.48 seconds)
2. **Missing Values Analysis**: Identified 67 columns with missing values (some >50% missing)
3. **Data Types**: 65 float64, 41 int64, 16 object columns
4. **Basic Exploration**: Target distribution, feature correlations, etc.

## Model Development Status
A credit risk modeling pipeline has been initiated with the following components:
- Data preprocessing module (missing value handling, encoding, scaling)
- Multiple ML algorithms comparison (Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM)
- Class imbalance handling (class weight balancing)
- Feature importance analysis
- Cross-validation framework
- Model saving/loading functionality

## Next Steps (if continuing)
1. Handle extreme missing values (>50% missing columns)
2. Feature selection/dimensionality reduction
3. Hyperparameter tuning for top-performing models
4. Ensemble methods
5. Model interpretation using SHAP values
6. Deployment preparation

## Files Created
- `explore_data.py`: Initial data exploration script
- `data_summary.txt`: Summary of dataset characteristics
- `credit_risk_model.py`: Complete modeling pipeline
- `test_load.py`, `test_imports.py`: Utility scripts for verification
- `training.log`: Training output log (when completed)

## Requirements
- Python 3.13+
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- matplotlib (for visualization)
- seaborn (for visualization)

## Usage
To run the complete modeling pipeline:
```python
python credit_risk_model.py
```

To load and use a saved model:
```python
from credit_risk_model import CreditRiskModel
model = CreditRiskModel()
model.load_model('credit_risk_model.pkl')
predictions, probabilities = model.predict(new_data)
```

## Notes
- The dataset is highly imbalanced (92% non-defaulters, 8% defaulters)
- Many features have significant missing values requiring careful handling
- Feature engineering opportunities exist in aggregating and transforming existing features
- Model evaluation should focus on precision-recall metrics and ROC-AUC due to class imbalance

## References
- Home Credit Default Risk Dataset: https://www.kaggle.com/c/home-credit-default-risk
- Feature descriptions available in HomeCredit_columns_description.txt (if included)