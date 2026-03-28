# Credit Card Default Prediction

This project predicts whether a customer will default on their credit card payment next month using machine learning. It is built as a binary classification workflow and includes exploratory data analysis, model training, evaluation, and a fairness-oriented review by gender.

## Project Overview

The notebook explores a well-known credit default dataset from the UCI Machine Learning Repository. It uses customer demographic and payment-history features to predict the target variable:

- `1` = Default
- `0` = No default

The workflow includes:
- data loading and inspection
- exploratory data analysis
- correlation analysis
- baseline modeling
- logistic regression
- random forest with hyperparameter tuning
- fairness checks across gender groups

## Dataset

The dataset contains customer credit information such as:

- `LIMIT_BAL` — credit limit
- `SEX` — gender
- `MARRIAGE` — marital status
- `AGE` — age
- `PAY_0` to `PAY_6` — repayment status history
- `BILL_AMT1` to `BILL_AMT6` — bill amounts
- `PAY_AMT1` to `PAY_AMT6` — payment amounts
- `default payment next month` — target label

## Key Findings

### Correlation analysis
The strongest positive relationship with default was found in recent repayment behavior, especially `PAY_0`. This suggests that recent payment status is one of the most important predictors of default risk.

### Fairness analysis
The notebook also checks whether default rates differ across gender groups:

- Female default rate: `0.207763`
- Male default rate: `0.241672`

This does not prove bias in the model by itself, but it does show that the dataset contains group-level differences that should be considered carefully.

## Models Used

### Baseline
A `DummyClassifier` using the most frequent class.

### Logistic Regression
An interpretable linear model trained with feature scaling and class balancing.

### Random Forest
A non-linear ensemble model tuned with `GridSearchCV`.

## Results

| Model | Accuracy |
| --- | ---: |
| Baseline (most frequent class) | 0.7790 |
| Logistic Regression | 0.6922 |
| Random Forest | 0.8135 |

Random Forest performed best in this notebook and achieved the highest test accuracy.

### Random Forest classification report
- Accuracy: `0.81`
- Precision for class 1 (default): `0.60`
- Recall for class 1 (default): `0.47`
- F1-score for class 1 (default): `0.53`

## Repository Structure

```text
.
├── Credit_Assignment.ipynb
├── README.md
└── default of credit card clients.csv   # dataset used in the notebook
```

## Requirements

The notebook uses the following Python libraries:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn

## Setup

Install the dependencies with:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## How to Run

1. Open the notebook in Jupyter Notebook, JupyterLab, or Google Colab.
2. Make sure the dataset CSV is available in the expected path.
3. Run the notebook cells from top to bottom.

If you are using Google Colab, update the dataset path in the notebook to match your Drive or upload location.

## Notes on Reproducibility

- The notebook uses `train_test_split(..., random_state=13)` for the main split.
- Random Forest tuning uses `GridSearchCV` with cross-validation.
- Some results may vary slightly depending on environment and package versions.

## Limitations

- Accuracy alone is not enough for a high-stakes credit default problem.
- The notebook includes a fairness check, but it is not a full fairness audit.
- The dataset may reflect historical or societal biases.
- Additional metrics such as ROC-AUC, precision-recall, calibration, and subgroup analysis would strengthen the evaluation.

## Future Improvements

Possible next steps include:

- adding ROC-AUC and precision-recall analysis
- testing threshold tuning
- comparing more models
- performing a deeper fairness audit
- improving feature engineering
- documenting deployment considerations

## Conclusion

This notebook demonstrates an end-to-end machine learning workflow for credit default prediction. The Random Forest model performs best on test accuracy, but fairness and real-world risk considerations mean the model should be used with caution.

## Acknowledgements

- UCI Machine Learning Repository
- scikit-learn
- pandas
- matplotlib
- seaborn
