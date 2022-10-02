# Lending Club Credit Risk Analysis

## Overview

Credit  risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. The purpose of this project is to use Supervised Machine Learning statistical algorithms to make predictions based on data patterns. We were tasked with using imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling.

Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company, we oversampled the data using the RandomOverSampler and  SMOTE algorithms. We then undersampled the data using the ClusterCentroids algorithm. Next, we used a combinatorial approach of over- and undersampling using the SMOTEENN algorithm. And finally, we compared the two new machine learning models that reduced bias, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk.  We also evaluated the preformance of these models and made a written recommendation on whether they should be used to predict credit risk.


### Resources

   * **Supervised Machine Learning was used to resample the LoanStats_2019Q1.csv dataset.**
   * **Python 3.7.9, Anaconda, Jupyter Notebook 6.4.12**
   * **Python Libraries: Scikit-learn 1.0.1 and imbalanced-learn 0.9.0 libraries** 
   * **credit_risk_resampling_starter_code.ipynb and credit_risk_ensemble_starter_code.ipynb**
   

## Results

The original dataset contained 115,675 loan applications in Q1 of 2019. We used the "loan status" to determine whether the application was considered "low" or "high" risk. Applications that had "current" as the "loan status" were classified as "low risk" and the remaining as "high risk". This reduced the dataset to 68,817 total applications with 99% classified as "low risk".

![datacount](https://github.com/rloufoster/Credit_Risk_Analysis/blob/main/Images/datacount.png?raw=true)

Using the 75/25% method to split the data for training vs. testing, 51,366 "low risk" and 246 "high risk" applications were categorized into the training set.   

![trainingdata](https://github.com/rloufoster/Credit_Risk_Analysis/blob/main/Images/trainingdata.png?raw=true)


### Deliverable 1 - Use Resampling Models to Predict Credit Risk

**Oversampling**

-RandomOverSampler randomly selects from the minority class and adds it to the training set until both classifications are equal.  The results classified **51,366** records each as High Risk and Low Risk.

![oversamplecount](https://github.com/rloufoster/Credit_Risk_Analysis/blob/main/Images/oversamplecount.png?raw=true)

   * Balanced accuracy score:  64%
   
   ![oversampleacc](https://github.com/rloufoster/Credit_Risk_Analysis/blob/main/Images/oversampleacc.png?raw=true)
   
   * The "High Risk" precision rate was only 1% with the recall at 66% giving this model an F1 score of 2%.
   * The "Low Risk" had a precision rate of 100% and recall of 62%.
   
   ![oversamplecm](https://github.com/rloufoster/Credit_Risk_Analysis/blob/main/Images/oversamplecm.png?raw=true)
   
   ![oversampleclass](https://github.com/rloufoster/Credit_Risk_Analysis/blob/main/Images/oversampleclass.png?raw=true)

-SMOTE (Synthetic Minority Oversampling Technique) Model, like RandomOverSampler, increases the size of the minority class by creating new values based on the value of the closest neighbors to the minority class instead of random selection.

   * Balanced accuracy score improved slightly to 65.1%.
   
   ![smoteacc](https://github.com/rloufoster/Credit_Risk_Analysis/blob/main/Images/Smoteacc.png?raw=true)
   
   * Like RandomOverSampler, the "High Risk" precision rate again was only 1% with the recall degraded to 61% giving this model an F1 score      of 2%.
   * The "Low Risk" had a precision rate of 100% and an improved recall at 69%.
   
   ![smotecm](https://github.com/rloufoster/Credit_Risk_Analysis/blob/main/Images/SmoteCM.png?raw=true)
   
   ![smoteclass](https://github.com/rloufoster/Credit_Risk_Analysis/blob/main/Images/SmoteClass.png?raw=true)
   

**Undersampling**

-ClusterCentroids Model is an algorithm that identifies clussters of the majority class to generate synthetic data points that are representative of the clusters.  The model classified 246 records each as High RIsk and Low Risk.

![undersamplecount](https://github.com/rloufoster/Credit_Risk_Analysis/blob/main/Images/undersamplecount.png?raw=true)

   * Balanced accuracy score was lower than the oversampling models at 54.5%.
   
   ![underacc](https://github.com/rloufoster/Credit_Risk_Analysis/blob/main/Images/underacc.png?raw=true)
   
   * The "High Risk" precision rate again was only at 1% with the recall at 69% giving this model an F1 score of 1%.
   * The "Low Risk" had a precision rte of 100% and with a lower recal at 40% compared to the oversampling models.
   
   ![undercm](https://github.com/rloufoster/Credit_Risk_Analysis/blob/main/Images/undercm.png?raw=true)
   
   ![underclass](https://github.com/rloufoster/Credit_Risk_Analysis/blob/main/Images/underclass.png?raw=true)
   
### Deliverable 2 - Use the SMOTEENN algorithm to Predict Credit Risk

**Combination Sampling**

-SMOTEENN (Synthetic Minority Oversampling Technique + Edited NearestNeighbors) Model is a combination of aspects of both oversampling and undersampling. The model classified **66,460** records as High Risk and **62,011** as Low Risk.

![SMOTEENNcount](https://github.com/rloufoster/Credit_Risk_Analysis/blob/main/Images/SMOTEENNcount.png?raw=true)

   * The balanced accuracy score improved to 64.5% when using the combined sampling model.
   
   ![SMOTEENNacc](https://github.com/rloufoster/Credit_Risk_Analysis/blob/main/Images/SMOTEENNacc.png?raw=true)
   
   * The "High Risk" precision rate did not improve at only 1%, however the recall increased to 72% giving this model an F1 score of 2%.
   * The "Low Risk" still showed a precision rate of 100% with the recall at 57%.
   
   ![SMOTEENNcm](https://github.com/rloufoster/Credit_Risk_Analysis/blob/main/Images/SMOTEENNcm.png?raw=true)
   
   ![SMOTEENNclass](https://github.com/rloufoster/Credit_Risk_Analysis/blob/main/Images/SMOTEENNclass.png?raw=true)
   

### Deliverable 3 - Use Ensemble Classifiers to Predict Credit Risk

Compare two new Machine Learning models that reduce bias to predict credit risk.  The models classified **51,366** as High Risk and 246 as Low Risk.

![Balancedcount](https://github.com/rloufoster/Credit_Risk_Analysis/blob/main/Images/balancedcount.png?raw=true)

-BalancedRandomForestClassifier Model, two decison trees of equal size to the minority class are created.  One to represent the majority class and the other to represent the minority class. 

   * The balanced accuracy score increased to 78.9% for this model.
   
   ![balanceacc](https://github.com/rloufoster/Credit_Risk_Analysis/blob/main/Images/Balancedacc.png?raw=true)
   
   * The "High Risk" precision rate increased to 3% with the recall at 70% giving this model an F1 score of 6%.
   * The "Low Risk" still had a precision rate of 100% with the recall at 87%.
   * The top feature by importance was "total_rec_prncp" at 7,9% of the total.
   
   ![balancecm](https://github.com/rloufoster/Credit_Risk_Analysis/blob/main/Images/Balancedcm.png?raw=true)
   
   ![balanceclass](https://github.com/rloufoster/Credit_Risk_Analysis/blob/main/Images/balancedclass.png?raw=true)
   
   ![balancefeature](https://github.com/rloufoster/Credit_Risk_Analysis/blob/main/Images/BalancedFeature.png?raw=true)
   
-EasyEnsembleClassifier Model is a set of classifiers where individual decision are combined to classify new examples.

   * The balanced accuracy score increased to 93.2% with this model.
   
   ![easyacc](https://github.com/rloufoster/Credit_Risk_Analysis/blob/main/Images/Easyacc.png?raw=true)
   
   * The "High Risk precision rate increased to 9% with the recall at 92% giving this model an F1 score of 16%.
   * The "Low Risk" still had a precision rate of 100% with the recall now at 94%.
   
   ![easycm](https://github.com/rloufoster/Credit_Risk_Analysis/blob/main/Images/Easycm.png?raw=true)
  
   ![easyclass](https://github.com/rloufoster/Credit_Risk_Analysis/blob/main/Images/Easyclass.png?raw=true)
   
   
## Summary

After exploring the six models, the EasyEnsembleClassifer model provided the best results with an accuracy rate of 93.2% and a 9% precision rate when predicting "High Risk" candidates. The recall rate was also the highest at 92% compared to the other models. The result for predicting "Low Risk" was also the highest with the recall rate at 94% and an F1 score of 97%. Therefore, if a model needed to be recommended to perform this type of analysis, this would be the apparent choice.

**Ranking of models in descending order based on "High Risk" results:**

* `EasyEnsembleClassifer`: 93.2% accuracy, 9% precision, 92% recall, and 16% F1 Score
* `BalancedRandomForestClassifer`: 78.9% accuracy, 3% precision, 70% recall and 6% F1 Score
* `SMOTE`: 65.2% accuracy, 1% precision, 61% recall and 2% F1 Score
* `SMOTEENN`: 64.5% accuracy, 1% precision, 72% recall and 2% F1 Score
* `RandomOverSampler`: 64.0% accuracy, 1% precision, 66% recall and 2% F1 Score
* `ClusterCentroids`: 54.5% accuracy, 1% precision, 69% recall and 1% F1 Score

   
   



   
   
   
   
   
   
   
