# Heart Disease Classification using Machine Learning

## Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to predict the presence of heart disease based on patient medical attributes. The project also demonstrates the deployment of trained models using a Streamlit web application.

## Dataset Description

The dataset contains 1026 records and 14 columns consisting of medical attributes such as age, sex, cholesterol, blood pressure, and other clinical parameters. The target variable indicates whether a patient has heart disease (1) or not (0). This is a binary classification problem.

## Models Used

The following machine learning models were implemented and evaluated:

* Logistic Regression
* Decision Tree Classifier
* K-Nearest Neighbors (KNN)
* Naive Bayes (Gaussian)
* Random Forest (Ensemble)
* XGBoost (Ensemble)

## Model Performance Comparison

|ML Model|Accuracy|AUC|Precision|Recall|F1|MCC|
|-|-|-|-|-|-|-|
|Logistic Regression|0.8098|0.9298|0.7619|0.9143|0.8312|0.6309|
|KNN|0.8634|0.9629|0.8738|0.8571|0.8654|0.7269|
|Naive Bayes|0.8293|0.9043|0.8070|0.8762|0.8402|0.6602|
|Decision Tree|0.9854|0.9857|1.0000|0.9714|0.9855|0.9712|
|Random Forest|1.0000|1.0000|1.0000|1.0000|1.0000|1.0000|
|XGBoost|1.0000|1.0000|1.0000|1.0000|1.0000|1.0000|

## Observations on Model Performance



|       ML Model           |                                                     Observation about model performance                                                     |

|--------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|

|   Logistic Regression    | Logistic Regression achieved reasonable performance with high recall, indicating it identifies most positive cases. However, its overall accuracy, F1-score, 				and MCC are lower compared to more complex models, suggesting limited capacity to capture non-linear patterns in the data.               |

|      Decision Tree       | The Decision Tree model produced very high performance with near-perfect metrics across most measures. This shows its ability to learn complex decision 					boundaries, though such strong results may indicate a tendency toward overfitting on this dataset.                                       |

|           KNN            | KNN showed balanced performance with good precision and recall and a solid F1-score and AUC. Its performance is better than simpler probabilistic models but 				still below tree-based ensemble methods.                                                                                                 |

|    Naive Bayes           | Naive Bayes delivered decent and stable results with moderate accuracy and MCC. Its simplifying assumption of feature independence likely limits its 					performance compared to more flexible models.                                                                                            |

| Random Forest (Ensemble) | Random Forest achieved perfect scores across all evaluation metrics, indicating excellent predictive performance and strong generalization on the test set. The 				ensemble nature helps reduce variance and improves robustness compared to a single decision tree.                                        |

|   XGBoost (Ensemble)     | XGBoost also achieved perfect scores across all metrics, matching Random Forest. This demonstrates the effectiveness of boosting-based ensemble methods in 				capturing complex patterns in the dataset and delivering top-tier performance.                                                           |

 

Overall, ensemble models (Random Forest and XGBoost) performed the best on this dataset, significantly outperforming simpler models such as Logistic Regression, KNN, and Naive Bayes.

