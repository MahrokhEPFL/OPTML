# OPTML
Code for the optimization for ML course report

Dataset:
Please download and extract halfhourly_dataset.zip from https://www.kaggle.com/jeanmidev/smart-meters-in-london and place in the inputs folder.

Notebooks:
- Visulaization: data cleaning, visualization, and exploratory data analysis
- Lin_Reg: linear regression to get started and familiarize with the data
- Mean-Reg;MTL: main notebook for the project

Files: 
- household: Python class for households, with model fitting and prediction methods
- feat_selection: recursive feature elimination and correlation-based feature selection, both with CV for tuning the number of features
- load_data: function that summarizes all steps for loading and cleaning data
- construct_dataset: code for creating regression matrices and dataframes
- analyze_model: various model evaluation measures and plotting tools