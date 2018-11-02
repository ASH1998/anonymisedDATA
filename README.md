# anonymisedDATA
[![Views](http://hits.dwyl.iof/ASH1998/anonymisedDATA.svg)](http://github.com/ASH1998/anonymisedDATA)

**Briefly describe the conceptual approach you chose! What are the trade-offs?**
  * First starting with [Exploratory Data Analysis](https://github.com/ASH1998/anonymisedDATA/blob/master/EDA.ipynb) : exploring both the test and train files.Getting the most important features out of `55 feature set`, and exploring them thoroughly. Veiwing the number of unique values to get how to deal with the `missing values`.
  * Training with different models and using `ensemble` with various algorithms. [This dir has all the experimental results.](https://github.com/ASH1998/anonymisedDATA/tree/master/experimental_models_nb) Models that I used are `MLPClassifier`, `KNeighborsClassifier`, `SVC`, `GaussianProcessClassifier`,`DecisionTreeClassifier`,  `AdaBoostClassifier`, `ExtraTreesClassifier`.
  * For all ensemble models and singely each algorithm gave `0.96` accuracy which is absurd due to  **Imbalance Class issue** i.e class 1 in the target value is roughly 0.3% of whole train data.
  * Then I tried `oversampling`, and `undersampling` and `ensemble sampling`, which resulted in no imbalance class issue, but now the models had bad accuracy including the ensemble ones.
  * Next I tried [Catboost](https://github.com/catboost/catboost) and lightGBM and xgboost, from which Catboost had the best accuracy.
  * The only trade-off is computational time, which is 10 mins on P100 GPU.
  
  **What's the model performance? What is the complexity? Where are the bottlenecks?**
  * Model is `CATBOOST`, a fast, scalable and high performance Gradient Boosting on Decision Trees. As it has GPU support, training was super fast too.
  * Metrics
  
  ![met](https://github.com/ASH1998/anonymisedDATA/blob/master/images/met.PNG)
  * On a standard laptop training for 20k iterations on `CPU` will take 10-15 mins with 95% accuracy.
  * On `GPU` training for 20k iterations takes 4-6 mins. Prediction time is <~0.1 sec on the test set.
  * Bottleneck : Transfering the same model, if we train a small dataset with the same parameters, performance will be slower than lightgbm and xgboost.
  


**If you had more time, what improvements would you make, and in what order of priority?**
  * PCA visualization.
  * Try voting classifier of deep catboost, lightgbm and xgboost(which will be computational expensive)
  * Write unit testing functions for the data preprocessing part.


![finalboard](https://github.com/ASH1998/anonymisedDATA/blob/master/images/final.PNG)

