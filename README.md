# Insurance Charges Prediction Project

## Overview

This project aims to predict insurance charges based on various factors using machine learning models. The dataset used contains information such as age, sex, BMI, number of children, smoking status, region, and insurance charges. The goal is to explore the data, perform data preprocessing, and build regression models to predict insurance charges.

## Project Structure

1. **Data Collection:**
    - The dataset is obtained from [source link](https://raw.githubusercontent.com/alexjolly28/entri_DSML/main/resources/insurance.csv) using the `wget` command.
    - The data is read into a Pandas DataFrame for further analysis.

    ```python
    # Downloading the data from the provided URL and reading it into a DataFrame
    !wget https://raw.githubusercontent.com/alexjolly28/entri_DSML/main/resources/insurance.csv
    df = pd.read_csv("insurance.csv")
    ```

2. **Exploratory Data Analysis (EDA):**
    - Basic libraries such as NumPy, Pandas, Matplotlib, and Seaborn are imported.
    - Descriptive statistics, null checks, and visualizations are performed to understand the data.

    ```python
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Descriptive stats
    df.describe()

    # Null check
    df.isnull().value_counts()

    # Age vs Charges visualization
    sns.lmplot(x='age', y='charges', data=df, scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
    plt.title('Age vs Insurance Charges')
    plt.xlabel('Age')
    plt.ylabel('Insurance Charges')
    plt.show()
    ```

3. **Data Preprocessing:**
    - Label encoding is applied to categorical variables (e.g., sex, smoker, region) to convert them into numerical format.

    ```python
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    df['sex_encoded'] = label_encoder.fit_transform(df['sex'])
    df['smoker_encoded'] = label_encoder.fit_transform(df['smoker'])
    df['region_encoded'] = label_encoder.fit_transform(df['region'])
    ```

    - Unnecessary columns are removed.

    ```python
    unnecessary_columns = ['sex', 'smoker', 'region']
    df = df.drop(columns=unnecessary_columns)
    ```

4. **Feature Selection:**
    - Independent (features) and dependent (target) variables are selected.

    ```python
    X = df[['age', 'sex_encoded', 'bmi', 'children', 'smoker_encoded', 'region_encoded']]
    y = df['charges']
    ```

5. **Modeling:**
    - Various regression models are explored, including Linear Regression, Decision Tree, Random Forest, and Support Vector Machine (SVR).

    ```python
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.svm import SVR

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initializing models
    models = [
        ('Linear Regression', LinearRegression()),
        ('Decision Tree', DecisionTreeRegressor(random_state=42)),
        ('Random Forest', RandomForestRegressor(n_estimators=100, random_state=42)),
        ('Support Vector Machine', SVR(kernel='linear'))
    ]

    # Evaluating models
    for model_name, model in models:
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        avg_mse = -cv_scores.mean()
        print(f"{model_name} - Average MSE: {avg_mse}")
    ```

6. **Conclusion:**
    - The model with the lowest average Mean Squared Error (MSE) can be considered the best model for predicting insurance charges.

---
