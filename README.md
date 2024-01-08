# Insurance Charges Prediction Project

## Overview

This project aims to predict insurance charges based on various factors using machine learning models. The dataset used contains information such as age, sex, BMI, number of children, smoking status, region, and insurance charges. The goal is to explore the data, perform data preprocessing, and build regression models to predict insurance charges.

## Project Structure

1. **Data Collection:**
    - The dataset is obtained from [source link](https://raw.githubusercontent.com/alexjolly28/entri_DSML/main/resources/insurance.csv) using the `wget` command.
    - The data is read into a Pandas DataFrame for further analysis.

2. **Exploratory Data Analysis (EDA):**
    - Basic libraries such as NumPy, Pandas, Matplotlib, and Seaborn are imported.
    - Descriptive statistics, null checks, and visualizations are performed to understand the data.

3. **Data Preprocessing:**
    - Label encoding is applied to categorical variables (e.g., sex, smoker, region) to convert them into numerical format.

4. **Feature Selection:**
    - Independent (features) and dependent (target) variables are selected.

5. **Modeling:**
    - Various regression models are explored, including Linear Regression, Decision Tree, Random Forest, and Support Vector Machine (SVR).
