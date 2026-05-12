# HBV-HCC Early Prediction Using Gene Expression and Machine Learning

**Final Year Project: BSc Computer Science**

## Overview
This project builds and compares machine learning models (Random Forest, SVM, XGBoost) 
to predict Hepatocellular Carcinoma (HCC) progression in Hepatitis B Virus (HBV) 
infected patients using gene expression data.

## Datasets
- GSE14520 The primary dataset (HBV-HCC, liver tissue)
- GSE236281 This will be used for cross-cohort validation (Nigerian cohort, PBMC)

## Project Structure
- `data/`: Raw and processed datasets
- `notebooks/`: Jupyter notebooks for each pipeline stage
- `models/`: Saved trained models
- `results/`: Figures, evaluation metrics, plots
- `docs/`: Report and documentation

## Pipeline
1. Data preprocessing & EDA
2. Differential expression analysis (feature selection)
3. Model training & hyperparameter tuning
4. Evaluation & comparison
5. Cross-cohort validation
