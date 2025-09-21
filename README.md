# MBR-XGB on MTO

## config
* './da_config_gbdt.json', './da_config_xgb.json', and './da_config_mbr.json'define the parameter bounds for the DA hyperparameter tuning of the GBDT, XGBoost, and MBR-XGB models, respectively.
* './models_config.json' contains the parameters for all models used in our training.

## Exploration
* './Exploration.ipynb' contains the preprocessing of the dataset and the exploratory data analysis. The plotting programs for Figures 2, 3, and 5 in the manuscript can be found here.

## KDE_plot
* './KDE_plot.R' contains the process of drawing kernel density distribution plots in R. The plotting program for Figure 4 in the manuscript can be found here.

## Functions
* './Functions.py' contains the functions required for data processing, model implementation, and plotting.

## ML_train
* './ML_train.ipynb' contains the training processes of the models and the saving of results. The plotting program for Figure 7 in the manuscript can be found here.

## DA for MBR_XGB
* './DA for MBR_XGB.ipynb' contains the process of optimizing the hyperparameters of the MBR_XGB model using the Dragonfly Algorithm.

## STT_test
* './STT_test.ipynb' contains the operation of the MBR-XGB model on the STT test set and the plotting program for Figure 8 in the manuscript.

## SHAP for MBR_XGB
* './SHAP for MBR_XGB.ipynb' contains the SHAP analysis process for MBR-XGB and the plotting program for Figure 9 in the manuscript.

## requirements
* './requirements.txt' specifies all Python packages and their respective versions utilized in this study.
