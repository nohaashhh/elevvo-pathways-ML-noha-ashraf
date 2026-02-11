# elevvo-pathways-ML-noha-ashraf
Machine learning Tasks
This repository contains --- machine learning projects demonstrating supervised, unsupervised, and tree-based modeling, implemented without relying on sklearn for core algorithms. The projects showcase data cleaning, feature engineering, model building, evaluation, and visualization.


## Task 1: Student Score Prediction

### Objective:
Predict students’ exam scores based on their study habits and other behavioral factors.

### Dataset:
Student Performance Factors (Kaggle)

### Methodology:

1-Data cleaning and handling missing values.

2-Basic visual exploration of features like study hours, sleep, participation.

3-Manual train-test split using NumPy/Pandas.

4-Built linear regression and polynomial regression models from scratch using NumPy.

5-Evaluated models with MAE, MSE, and R², manually calculated.

6-Experimented with different feature combinations to analyze the impact on predictions.

### Results:

-Polynomial regression captured non-linear relationships better than simple linear regression.

-Including additional features like sleep hours and participation improved prediction accuracy.

Visualizations show predicted vs actual scores and residuals.

### Key Insights:

-Study hours have the largest influence on exam performance.

-Sleep and participation are significant secondary factors.

### Added in this task:
-GUI for predicting student's score according to his attendance % , Hours studed and previous scores.


## Task 2: Customer Segmentation

### Objective:
Cluster mall customers based on their annual income and spending behavior to identify distinct customer segments.

### Dataset:
Mall Customers (Kaggle)

### Methodology:

1-Selected Annual Income and Spending Score as features.

2-Scaled features manually using standardization (mean 0, std 1).

3-Visual exploration of customer distribution.

4-Implemented K-Means clustering from scratch using NumPy.

5-Determined optimal number of clusters using manual elbow method.

6-Visualized clusters and centroids in 2D.

### Bonus:

-Implemented DBSCAN for density-based clustering.

-Analyzed average spending per cluster for actionable insights.

### Results:

Identified 4–5 clear customer segments based on spending behavior.

DBSCAN revealed potential outliers and dense customer groups.

Average spending per cluster helped identify premium vs low-value customers.

### Key Insights:

High-income, high-spending customers form a distinct cluster suitable for targeted campaigns.

Low-income but high-spending customers may benefit from loyalty programs.



## Task 3: Forest Cover Type Classification

### Objective:
Predict forest cover types based on cartographic and environmental features.

### Dataset:
Covertype (UCI)

### Methodology:

1-Data cleaning and preprocessing; no categorical encoding required since features are one-hot encoded.

2-Manual train-test split using NumPy/Pandas.

3-Trained XGBoost multi-class classifier for predicting cover type.

4-Evaluated model using accuracy and a manual confusion matrix.

5-Analyzed feature importance using XGBoost’s gain metric.

### Bonus: 

Compared with Random Forest–style model in XGBoost.

Performed manual hyperparameter tuning to improve performance.

### Results:

High accuracy achieved (~85–90%) without sklearn.

Confusion matrix shows most misclassifications occur between similar terrain types.

Feature importance reveals Elevation, Horizontal Distances, and Soil Types as key predictors.

### Key Insights:

Elevation and proximity to water/roads dominate classification decisions.

Tree-based models handle categorical one-hot features and continuous features effectively.


## Task 4: Loan Approval Prediction

### Objective

Predict whether a loan application will be approved or rejected based on applicant data. The goal is to assist banks and financial institutions in automating and improving loan decision-making, 
while minimizing risk. Key focus is on precision, recall, and F1-score due to class imbalance.

### Dataset

Source: Loan-Approval-Prediction-Dataset (Kaggle)

### Methodology

1-Data Preprocessing

2-Handle missing values using mean/median for numerical features and mode for categorical features

3-Encode categorical variables (one-hot encoding for multi-class, label encoding for binary)

4-Combine income features: TotalIncome = ApplicantIncome + CoapplicantIncome

5-Apply log transformation to skewed features (LoanAmount, TotalIncome)

6-Handling Imbalanced Data

7-Apply SMOTE to oversample minority class (approved/rejected loans)

8-Model Selection & Training

9-Logistic Regression for interpretability

10-Decision Tree and Random Forest for tree-based modeling

11-Train/test split (80/20)

12-Model Evaluation

13-Confusion matrix visualization

14-Precision, Recall, and F1-score as primary metrics

15-Compare model performance and choose the best classifier

### Results

Tree-based models slightly outperform logistic regression in all metrics, but logistic regression provides better interpretability for decision-making.

### key insights

Applicants with high total income and good credit history have a higher probability of loan approval

Property area and education also influence approvals, with urban and graduate applicants more likely to get approved

Imbalanced datasets require oversampling techniques like SMOTE to improve minority class predictions

Tree-based models can capture non-linear relationships, while logistic regression offers clear decision rules for approval

## General Notes

All projects implemented without sklearn, using NumPy and Pandas for data handling, manual train-test splits, evaluation metrics, and algorithm implementations. (SKLEARN GIVE ME ERRORS)

Visualizations created using Matplotlib to demonstrate insights and model performance.

Bonus experiments include polynomial regression, DBSCAN clustering, Random Forest comparison, and hyperparameter tuning.
