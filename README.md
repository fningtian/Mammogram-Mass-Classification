# Mammogram-Mass-Classification
Final project from **Machine Learning, Data Science and Deep Learning with Python** by Udemy [**[Certificate](https://www.udemy.com/certificate/UC-695fd1d5-39f9-426a-920f-3d2d53c20773/)**]

A lot of unnecessary anguish and surgery arises from false positives arising from mammogram results. If we can build a better way to interpret them through supervised machine learning, it could improve a lot of lives. **In this project, the goal is to predict the severity (benign or malignant) of a mammographic mass lesion by applying supervised machine learning tenichques and neural networks.**

## The dataset
[Mammographic Masses](https://archive.ics.uci.edu/ml/datasets/Mammographic+Mass) public dataset from the UCI repository

This dataset contains 961 instances of masses detected in mammograms and the following attributes:

| Attributes | Description | Data Types |
| ------ | ------ | ------ |
| BI-RADS | 1 to 5 | ordinal |
| Age | patient's age in years | integer |
| Shape | round=1 oval=2 lobular=3 irregular=4 | nominal |
| Margin | circumscribed=1, microlobulated=2, obscured=3, ill-defined=4, spiculated=5 | nominal | 
| Density | high=1, iso=2, low=3, fat-containing=4 | ordinal |
| Severity | benign=0 or malignant=1 | binominal | 

BI-RADS is an assesment of how confident the severity classification is; it is not a "predictive" attribute. The age, shape, margin, and density attributes are the features to build the model with, and "severity" is the classification to predict based on those attributes. 

> Note: 
Although "shape" and "margin" are nominal data types, which sklearn typically doesn't deal with well, they are close enough to ordinal. The "shape" for example is ordered increasingly from round to irregular.

## Algorithms and hyperparameters

Supervised Learning
- Logistic Regression - penalty, C
- Decision tree 
- Random forest - n_estimators, max_features, bootstrap
- KNN - knn__n_neighbors
- Naive Bayes
- SVM - kernel, C

Deep Learning
- Neural network using Keras

## Performance evaluation
The performance metric is accuracy measured with K-Fold class validation.

## Requirements
The required libraries include numpy, pandas, matplotlib, seaborn, sklearn, tensorflow and keras.

## Results
The performance comparison is docomented in the Jupyter Notebook. 

## About the [course](https://www.udemy.com/course/data-science-and-machine-learning-with-python-hands-on/learn/lecture/14315154#overview)
The course covers the following topics
- Build artificial neural networks with Tensorflow and Keras
- Classify images, data, and sentiments using deep learning
- Make predictions using linear regression, polynomial regression, and multivariate regression
- Data Visualization with MatPlotLib and Seaborn
- Implement machine learning at massive scale with Apache Spark's MLLib
- Understand reinforcement learning - and how to build a Pac-Man bot
- Classify data using K-Means clustering, Support Vector Machines (SVM), KNN, Decision Trees, Naive Bayes, and PCA
- Use train/test and K-Fold cross validation to choose and tune your models
- Build a movie recommender system using item-based and user-based collaborative filtering
- Clean your input data to remove outliers
- Design and evaluate A/B tests using T-Tests and P-Values
