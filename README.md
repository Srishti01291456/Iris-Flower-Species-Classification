## 🌸Iris Flower Species Classification using Machine Learning
### 📌 Project Overview

This project aims to classify Iris flowers into three species — Setosa, Versicolor, and Virginica — using Machine Learning techniques. The project follows a complete ML workflow including data preprocessing, exploratory data analysis (EDA), model training, hyperparameter tuning, evaluation, and deployment through a Streamlit web application.

The goal is to build a robust classification model that can accurately predict the species of an Iris flower based on its sepal and petal measurements.

### 📂 Dataset Information

The Iris dataset is one of the most widely used datasets in Machine Learning.

# Dataset Features:

Feature	Description
Sepal Length	Length of sepal in cm
Sepal Width	Width of sepal in cm
Petal Length	Length of petal in cm
Petal Width	Width of petal in cm

# Target Classes:

Iris Setosa
Iris Versicolor
Iris Virginica

# Dataset Statistics:

Total Samples: 150
Features: 4
Classes: 3
# 🛠 Technologies Used
Programming Language
Python
Libraries
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
Pickle
Deployment
Streamlit
Development Tools
Jupyter Notebook
VS Code
Git
GitHub
# 📊 Exploratory Data Analysis

Performed detailed EDA to understand dataset characteristics and relationships between features.

Analysis Performed

✔ Checked dataset structure and summary statistics

✔ Verified missing values and duplicates

✔ Visualized feature distributions

✔ Generated pair plots for class separation analysis

✔ Created correlation heatmap

✔ Studied feature relationships among different species

### Key Findings
Setosa species is easily separable from the other classes.
Petal length and petal width are highly informative features.
Dataset contains no missing values.
Features show strong correlation patterns useful for classification.
### 🤖 Machine Learning Models

The following classification algorithms were implemented and compared:

1. Logistic Regression

A baseline linear classification model.

2. K-Nearest Neighbors (KNN)

Classifies samples based on nearest neighbors.

3. Random Forest Classifier

Ensemble learning model using multiple decision trees.

Model Comparison
Model	Accuracy
Logistic Regression	XX%
KNN	XX%
Random Forest	XX%

(Replace XX with your actual results.)

### ⚙ Hyperparameter Tuning

To improve model performance, GridSearchCV was used for hyperparameter optimization.

Parameters Tuned
KNN
n_neighbors
weights
metric
Random Forest
n_estimators
max_depth
min_samples_split
Benefits
Improved model accuracy
Reduced overfitting
Selected optimal parameter combinations
### 📈 Model Evaluation

The models were evaluated using multiple performance metrics.

Evaluation Metrics
Accuracy Score
Confusion Matrix
Classification Report
Cross Validation
Results
Accuracy Score: XX%
Cross Validation Score: XX%
Evaluation Visualizations
Confusion Matrix Heatmap
Accuracy Comparison Chart

The Random Forest model achieved the best performance and was selected for deployment.

### 🚀 Streamlit Deployment

A user-friendly Streamlit web application was developed to allow real-time predictions.

Features

✅ Interactive input fields

✅ Real-time species prediction

✅ Clean and responsive interface

✅ Instant prediction results

### User Inputs
Sepal Length
Sepal Width
Petal Length
Petal Width
Output

### Predicted Iris Flower Species

# 📸 Screenshots
Application Home Page
<img width="1328" height="533" alt="image" src="https://github.com/user-attachments/assets/7a9d4119-f986-4248-ac00-3185c41bdbf2" />

Prediction Result
<img width="940" height="436" alt="image" src="https://github.com/user-attachments/assets/6be8560b-b374-4519-a976-afcc1c0d6dc5" />
<img width="855" height="597" alt="image" src="https://github.com/user-attachments/assets/073cf162-6120-4553-90cb-65a1058260dc" />
<img width="894" height="611" alt="image" src="https://github.com/user-attachments/assets/b9edca52-50d8-4638-a1e8-05983d49e58b" />
<img width="924" height="458" alt="image" src="https://github.com/user-attachments/assets/dbf0ba19-5af6-4b25-9d13-07c740784ba2" />
<img width="429" height="186" alt="image" src="https://github.com/user-attachments/assets/252fbcbd-903c-4090-8bbc-8f9460b720c1" />


### ▶ How to Run
Clone Repository
git clone https://github.com/yourusername/Iris-Flower-Classification.git
Navigate to Project
cd Iris-Flower-Classification
Install Dependencies
pip install -r requirements.txt
Run Streamlit Application
streamlit run app.py
Open Browser
http://localhost:8501
### 📌 Future Improvements
Deploy application on Streamlit Cloud.
Add probability scores for predictions.
Implement additional classification algorithms.
Add advanced visualizations and dashboards.
Enable batch prediction using CSV upload.
Containerize application using Docker.
Integrate CI/CD pipeline for automated deployment.
