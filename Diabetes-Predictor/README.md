# Diabetes Predictor
A simple machine learning project using logistic regression to predict diabetes based on medical measurements.


#  Project Overview

This project aims to **predict whether a person has diabetes** using medical diagnostic measurements. We use a supervised machine learning approach and the **Logistic Regression** algorithm to classify individuals as diabetic or not, based on features like glucose levels, BMI, and age.

Dataset used: **Pima Indians Diabetes Dataset** from the UCI Machine Learning Repository.


## Objective

To build a machine learning model that:
- Learns from medical data of individuals
- Predicts the **binary outcome** (1 = diabetic, 0 = non-diabetic)
- Provides performance metrics to evaluate prediction quality


## Project Structure

diabetes-predictor/
│
├── diabetes.csv              # Dataset file
├── main.py                   # Python script to train and evaluate model
├── README.md                 # Project notes and documentation



## Tools & Libraries Used

- **Python 3**
- **Pandas** – data manipulation
- **NumPy** – numerical operations
- **Matplotlib & Seaborn** – visualization
- **scikit-learn** – machine learning toolkit



## Dataset Description

The dataset consists of **768 rows** and **9 columns**:

| Column Name              | Description                                |
|--------------------------|--------------------------------------------|
| Pregnancies              | Number of times pregnant                   |
| Glucose                  | Plasma glucose concentration               |
| BloodPressure            | Diastolic blood pressure                   |
| SkinThickness            | Triceps skin fold thickness                |
| Insulin                  | 2-Hour serum insulin                       |
| BMI                      | Body mass index                            |
| DiabetesPedigreeFunction | Diabetes pedigree function (genetic risk) |
| Age                      | Age in years                               |
| Outcome                  | 1 = Diabetic, 0 = Not diabetic             |


##  Steps We Followed

### 1. **Setting Up the Project**
- Created a new folder in **VS Code**
- Installed necessary libraries using `pip`

### 2. **Downloading the Dataset**
- Retrieved from: [Pima Indians Dataset](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)
- Saved as `diabetes.csv` in the project folder

### 3. **Loading & Exploring the Data**
- Used `pandas` to load the CSV
- Added column names for better readability
- Previewed data using `.head()`, `.info()`, `.describe()`

### 4. **Data Preparation**
- Separated **features (X)** and **labels (y)**
- Split data into **training (80%)** and **testing (20%)**

### 5. **Model Training**
- Chose **Logistic Regression** – good for binary classification
- Trained the model on training data using `.fit()`

### 6. **Prediction & Evaluation**
- Predicted using `.predict()` on test data
- Evaluated using:
  - **Accuracy score**
  - **Confusion matrix**
  - **Classification report**


## Model Output

Example output from the model:

Accuracy: 0.79

Classification Report:
              precision    recall  f1-score   support

           0       0.83      0.88      0.85       100
           1       0.71      0.62      0.66        54

    accuracy                           0.79       154
   macro avg       0.77      0.75      0.76       154
weighted avg       0.78      0.79      0.78       154

Confusion Matrix:
[[88 12]
 [21 33]]

- **Accuracy**: 79%
- **Precision for diabetic (1)**: 71%
- **Recall for diabetic (1)**: 62%


## Conclusion

- The model successfully learned patterns to predict diabetes with decent accuracy.
- Logistic Regression provided a simple but effective baseline.
- Can be improved with more features, feature engineering, and better models.


## Future Improvements

- Try different models: **Random Forest**, **XGBoost**, **SVM**
- Handle missing/zero values in features like `Insulin` and `SkinThickness`
- Scale/normalize data for better performance
- Hyperparameter tuning with `GridSearchCV`
- Build a **web interface** using Flask/Streamlit for user input and prediction