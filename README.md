# anonymisedDATA
[![Views](http://hits.dwyl.io/ASH1998/anonymisedDATA.svg)](http://github.com/ASH1998/anonymisedDATA) 

## Dependencies
1. Python3
2. Sklearn
3. Matplotlib
4. Pandas
5. Catboost, xgboost, lightgbm, sklearn
6. tqdm

## Repo structure [![badge2](https://img.shields.io/badge/repo-download-blue.svg)](https://github.com/ASH1998/anonymisedDATA/archive/master.zip)
1. **[model.py](https://github.com/ASH1998/anonymisedDATA/blob/master/model.py)** : The python script for the catboost model generates the result csv.
2. **[finalsub.csv](https://github.com/ASH1998/anonymisedDATA/blob/master/finalsub.csv)** : The submission file. [![badge1](https://img.shields.io/badge/SubmissionCSV-view%20raw-brightgreen.svg)](https://github.com/ASH1998/anonymisedDATA/raw/master/finalsub.csv) 
3. [main.ipynb](https://github.com/ASH1998/anonymisedDATA/blob/master/main.ipynb) : notebook for model.py
4. [download.py](https://github.com/ASH1998/anonymisedDATA/blob/master/download.py) : Download the data
5. **[zipfilee_FILES/ds_data](https://github.com/ASH1998/anonymisedDATA/tree/master/zipfilee_FILES/ds_data)** : train and test csv files to be stored here.
6. [EDA.ipynb](https://github.com/ASH1998/anonymisedDATA/blob/master/EDA.ipynb) : Exploratory Data Analysis of the data.
7. [oversampling](https://github.com/ASH1998/anonymisedDATA/blob/master/oversampling.ipynb) : Experiments on Oversampling.
8. [experimental_model_nb dir](https://github.com/ASH1998/anonymisedDATA/tree/master/experimental_models_nb) : Experiments on data
9. [catboost_info](https://github.com/ASH1998/anonymisedDATA/tree/master/catboost_info) : model storage, model graphs(tensorboard files)
 10. [Images](https://github.com/ASH1998/anonymisedDATA/tree/master/images) Images from eda for readme file.
 

## Usage
1. Run on terminal/bash `python3 downloads.py`. It downloads the data. Extract and move the `test.csv` and `train.csv` into `zipfilee_FILES/ds_data`.
2. Run the `model.py` by `python3 model.py`. It will show the results. Uncomment the last lines to generate the `result.csv` for test data.
3. Same with rest *.ipynb files*. Using `jupyter notebook --allow-root` 


## EDA
1. Feature importance

![import](https://github.com/ASH1998/anonymisedDATA/blob/master/images/importance.PNG)

2. Box plots to see feature variance

![finalboard](https://github.com/ASH1998/anonymisedDATA/blob/master/images/der.PNG)

3. Distribution of target values

![val](https://github.com/ASH1998/anonymisedDATA/blob/master/images/sactter.PNG)

4. Train-Test-Validation Split
![split](https://github.com/ASH1998/anonymisedDATA/blob/master/images/tt.PNG)

5. After training :
 A. `100000 iterations` with learning_rate=0.01
 
 ![100](https://github.com/ASH1998/anonymisedDATA/blob/master/images/100k.PNG)
 
 B. `20000 iterations` with learning_rate=0.3 (best)
 
 ![20](https://github.com/ASH1998/anonymisedDATA/blob/master/images/20k.PNG)

## Questions
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


### FINAL OUTPUT on tensorboard.
![der](https://github.com/ASH1998/anonymisedDATA/blob/master/images/final.PNG)
