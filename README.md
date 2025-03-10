# Credit Risk Analysis and Scoring Model

## Project Overview
This project implements a credit risk analysis and scoring model using customer data and CIC (Credit Information Corporation) data. The model predicts the likelihood of first payment default (FPD10+) and creates a scoring mechanism for credit risk assessment.

## Features
- Data processing and cleaning of complex nested CIC data
- Feature extraction and selection for predictive modeling
- Credit risk scorecard development
- Model performance evaluation using AUC-ROC and Gini coefficient
- Population Stability Index (PSI) analysis
- Score binning and distribution analysis

## Repository Structure
- `Report.ipynb`: Main Jupyter notebook with full analysis
- `functions.py`: Custom utility functions for data processing
- `bao.py`: Custom library for model development
- `scorecard.xlsx`: Generated scorecard with feature transformations
- `full_data.xlsx`: Processed dataset with extracted features
- `train_data.xlsx`/`test_data.xlsx`: Train/test split datasets

## Requirements
- Python 3.8+
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- calamine (for Excel processing)

## Results
The model achieves significant predictive power for credit default risk, as demonstrated by:
- Gini coefficient analysis
- Kolmogorov-Smirnov statistics
- AUC-ROC curve evaluation

## Usage
Open the `Report.ipynb` notebook to see the full analysis and model development process.
