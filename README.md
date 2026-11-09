# AMEX Default Prediction

A notebook-based study of credit default prediction using the [Kaggle American Express – Default Prediction competition](https://www.kaggle.com/competitions/amex-default-prediction). The workflow covers data compression, exploratory analysis, customer-level aggregation, temporal feature engineering, model comparison, and LightGBM tuning and validation.

## Data source and reproduction

The training data comes from the official [Kaggle competition data page](https://www.kaggle.com/competitions/amex-default-prediction/data). Input datasets are not included in this repository. To reproduce the experiments:

1. Download and extract **`train_data.csv`** and **`train_labels.csv`** from the official page.
2. Create a folder named **`data`** in the project root and place the two files inside it:

   ```text
   AMEX/
   ├── README.md
   ├── data/
   │   ├── train_data.csv
   │   └── train_labels.csv
   ├── 00_compression/
   ├── 01_overview/
   └── ...
   ```

3. Install the Python packages used by the notebooks:

   ```bash
   python -m pip install jupyterlab numpy pandas pyarrow matplotlib seaborn scikit-learn scipy lightgbm joblib
   ```

4. Run notebook **00** first to generate `data/train_data_float16_zstd.feather`. Then run the remaining notebooks in the order below, executing each notebook's cells from top to bottom. Intermediate feature tables must be regenerated locally; saved models and reports do not replace those dependencies.


## Notebook walkthrough

| Step | Notebook | Purpose |
| --- | --- | --- |
| 00 | [Data compression](00_compression/00_data_compression.ipynb) | Streams the raw training CSV into typed, compressed Parquet, then converts continuous features to float16 and saves a compressed Feather file for downstream analysis. Checks schemas and row counts. |
| 01 | [Exploratory data analysis](01_overview/01_exploratory_data_analysis.ipynb) | Examines dataset size, customer history lengths, statement dates, feature types, missing values, and the target distribution. Establishes why modeling and validation must operate at customer level. |
| 02 | [Customer aggregation](02_aggregation/02_customer_aggregation.ipynb) | Converts monthly statements into one row per customer. Computes mean, standard deviation, minimum, maximum, and last values for continuous features; keeps last categorical values; joins the labels; and writes `train_aggregated.parquet`. |
| 03 | [LightGBM baseline](03_baseline/03_lightgbm_baseline.ipynb) | Trains an initial LightGBM model using the aggregated features and a stratified customer-level holdout. Evaluates performance and saves the model, feature importance, metrics, and validation predictions. |
| 04 | [Temporal features](04_temporal_features/04_lightgbm_temporal_features.ipynb) | Selects the top 25 original variables using baseline importance and adds temporal summaries: first value, last minus first, last minus mean, and recent three- and six-statement means. Compares the resulting model against the baseline under the same validation protocol. |
| 04.5 | [Temporal feature refinement](04.5_temporal_refined/04.5_lightgbm_temporal_features.ipynb) | Replaces each selected variable's `first` feature with `last_minus_recent3_mean`, keeping the overall feature count and experimental settings fixed. This refined table is the reference used by the current later experiments. |
| 05 | [Categorical history features](05_categorical_features/05_lightgbm_temporal_categorical_refine.ipynb) | Extends 04.5 with historical category counts, transition counts, and last-state-change indicators. This experiment did not deliver a meaningful overall improvement, so its features are not carried into the main workflow from 06 onward. |
| 06 | [Logistic regression vs. LightGBM](06_model_comparison/06_logistic_regression_vs_lightgbm.ipynb) | Uses the **04.5 feature table** to compare an SGD-trained logistic regression model against the saved LightGBM benchmark on identical validation customers. Applies training-only scaling, missing-value handling, and categorical encoding, then saves coefficients, metrics, and predictions. |
| 07 | [Baseline cross-validation](07_baseline_cv/07_lightgbm_04_5_cross_validation.ipynb) | Evaluates the 04.5 LightGBM configuration with stratified three-fold cross-validation. Saves fold models, metrics, feature importance, and out-of-fold predictions, establishing the fold assignment for later comparisons. |
| 08 | [Hyperparameter tuning](08_hyperparameter_tuning/08_lightgbm_hyperparameter_tuning.ipynb) | Keeps the 04.5 features fixed while testing LightGBM tree complexity, sampling, regularization, and histogram resolution on the established holdout. Records trials and nominates candidates for cross-validation. |
| 09 | [Tuned-model cross-validation](09_tuned_cv/09_lightgbm_tuned_cross_validation.ipynb) | Evaluates tuned candidates T11 and T8 using the same folds as 07. Compares fold and full out-of-fold performance against the reference configuration and selects a preferred candidate. It does not perform a final full-data fit or generate Kaggle test predictions. |
