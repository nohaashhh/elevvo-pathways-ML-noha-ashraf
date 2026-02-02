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

###Bonus: 

Compared with Random Forest–style model in XGBoost.

Performed manual hyperparameter tuning to improve performance.

### Results:

High accuracy achieved (~85–90%) without sklearn.

Confusion matrix shows most misclassifications occur between similar terrain types.

Feature importance reveals Elevation, Horizontal Distances, and Soil Types as key predictors.

### Key Insights:

Elevation and proximity to water/roads dominate classification decisions.

Tree-based models handle categorical one-hot features and continuous features effectively.

## General Notes

All projects implemented without sklearn, using NumPy and Pandas for data handling, manual train-test splits, evaluation metrics, and algorithm implementations. (SKLEARN GIVE ME ERRORS)

Visualizations created using Matplotlib to demonstrate insights and model performance.

Bonus experiments include polynomial regression, DBSCAN clustering, Random Forest comparison, and hyperparameter tuning.
