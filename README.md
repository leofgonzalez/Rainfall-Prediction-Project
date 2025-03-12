# Weather Data Analysis and Prediction

This project focuses on analyzing weather data and building predictive models to forecast whether it will rain the next day (`RainTomorrow`). The project uses various machine learning algorithms, including **Linear Regression**, **K-Nearest Neighbors (KNN)**, **Decision Trees**, **Logistic Regression**, and **Support Vector Machines (SVM)**, to predict rainfall based on historical weather data.

---

## Project Overview
The goal of this project is to predict whether it will rain tomorrow (`RainTomorrow`) based on historical weather data. The project involves:
- Data preprocessing and cleaning.
- Exploratory data analysis (EDA).
- Building and evaluating multiple machine learning models.
- Comparing the performance of different algorithms.

---

## Dataset
The dataset used in this project is **`Weather_Data.csv`**, which contains historical weather data for Sydney, Australia. It includes features such as:
- Temperature (`MinTemp`, `MaxTemp`).
- Rainfall (`Rainfall`).
- Humidity (`Humidity9am`, `Humidity3pm`).
- Wind speed and direction (`WindGustDir`, `WindSpeed9am`, `WindSpeed3pm`).
- Pressure (`Pressure9am`, `Pressure3pm`).
- Cloud cover (`Cloud9am`, `Cloud3pm`).
- Target variable: `RainTomorrow` (binary classification).

The dataset is sourced from the IBM Developer Skills Network. 
- LINK to access the dataset: https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillUp/labs/ML-FinalAssignment/Weather_Data.csv

---

## Algorithms Used

The following machine learning algorithms are implemented and evaluated in this project:

### Linear Regression
  - Used to predict the probability of rain
  - Evaluated using metrics like R² score, mean absolute error (MAE), and mean squared error (MSE)
### K-Nearest Neighbors (KNN):
  - Classifies whether it will rain based on the nearest neighbors
  - Evaluated using accuracy, Jaccard score, and F1 score
### Decision Trees:
  - Builds a tree-based model to classify rainfall
  - Evaluated using accuracy, Jaccard score, and F1 score
### Logistic Regression:
  - Predicts the probability of rain using a logistic function
  - Evaluated using accuracy, Jaccard score, F1 score, and log loss
### Support Vector Machines (SVM):
  - Classifies rainfall using an SVM with an RBF kernel
  - Evaluated using accuracy, F1 score, and Jaccard score

---

## Results

The performance of each algorithm is summarized below:
Algorithm	Accuracy	Jaccard Score	F1 Score	Log Loss	R² Score (Linear Regression)
Linear Regression	-	-	-	-	0.37
K-Nearest Neighbors (KNN)	0.824	0.417	0.588	-	-
Decision Trees	0.760	0.732	0.845	-	-
Logistic Regression	0.829	0.793	0.669	6.163	-
Support Vector Machines (SVM)	0.719	0.517	0.602	-	-

---

## Conclusion

This project successfully implemented and evaluated multiple machine learning algorithms to predict whether it will rain the next day (RainTomorrow) based on historical weather data from Sydney, Australia. The following key insights were derived from the analysis:

### Best Performing Model:
- Logistic Regression achieved the highest accuracy (82.9%) among all the models tested. It also demonstrated strong performance in terms of the Jaccard score (0.793) and F1 score (0.669), making it the most reliable model for this classification task.
### Other Models:
- K-Nearest Neighbors (KNN) performed well with an accuracy of 82.4% and an F1 score of 0.588, but it was slightly less consistent than Logistic Regression
- Decision Trees achieved an accuracy of 76.0% and the highest F1 score (0.845), indicating good precision and recall for the "No Rain" class
- Support Vector Machines (SVM) had the lowest accuracy (71.9%) among the classifiers, suggesting it may not be the best choice for this dataset
- Linear Regression was used for regression analysis, achieving an R² score of 0.37, which indicates moderate explanatory power for predicting the probability of rain
### Key Takeaways:
- The dataset's features, such as temperature, humidity, and wind speed, were effective in predicting rainfall
- Logistic Regression emerged as the most suitable model for this binary classification problem due to its balance of accuracy, interpretability, and computational efficiency
- While Decision Trees had the highest F1 score, their lower accuracy suggests they may overfit the training data or struggle with generalization

### Future Improvements:
- Experiment with more advanced algorithms like Random Forests, Gradient Boosting, or Neural Networks to improve performance
- Perform hyperparameter tuning to optimize the existing models
- Incorporate additional features or external data sources (e.g., satellite data) to enhance predictive power
