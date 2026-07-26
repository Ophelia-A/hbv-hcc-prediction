# HBV-HCC Early Prediction Using Gene Expression and Machine Learning

**Final Year Project: BSc Computer Science**

## Overview
This project builds and compares machine learning models (Random Forest, SVM, XGBoost) 
to classify liver tissue samples as tumor or non-tumor in Hepatitis B Virus (HBV) 
infected patients, using Affymetrix GPL570 microarray gene expression data. The final 
model is deployed in a web application that allows a user to upload gene expression 
data and receive a prediction with confidence score and SHAP-based interpretability.

## Datasets
- **GSE14520**: Primary dataset (HBV-HCC, liver tissue) used for training and internal evaluation
- **GSE84402**: External validation cohort (28 HBV-HCC samples, same GPL570 platform)
- **GSE236281**: Investigated as a potential Nigerian cohort validation set, but contained 
  only DESeq2 summary statistics rather than per-sample expression matrices, which 
  precluded direct validation. Documented as a limitation reflecting the scarcity of 
  African genomic data in public repositories, rather than falsely resolved.

## Project Structure
- `data/`: Raw and processed datasets
- `notebooks/`: Jupyter notebooks for each pipeline stage
- `models/`: Saved trained models
- `results/`: Figures, evaluation metrics, plots
- `docs/`: Report and documentation
- `app/`: Flask web application for interactive prediction

## Pipeline
1. Data preprocessing & EDA
2. Differential expression analysis (feature selection)
3. LASSO feature selection (71 probes selected)
4. Model training & hyperparameter tuning (Random Forest, XGBoost, SVM)
5. Evaluation & model comparison (XGBoost selected as best model, F1-score: 0.9778)
6. External validation on GSE84402
7. SHAP interpretability analysis

## Web Application
A Flask-based web app allows a user to upload gene expression data (CSV, 71 LASSO-selected 
probes) and receive:
- A tumor / non-tumor prediction with confidence score, using the trained XGBoost model
- A visual "probe strip" showing this sample's expression levels across all 71 genes, 
  relative to training-set averages
- Top contributing genes for this specific prediction, via SHAP (TreeExplainer)
To run locally:
cd app
python app.py
Then open `http://127.0.0.1:5000` in a browser.