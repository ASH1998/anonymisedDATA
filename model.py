#imports
import matplotlib as mat
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale, LabelEncoder
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error, mean_squared_error, log_loss
from catboost import CatBoostClassifier, Pool

from imblearn.over_sampling import RandomOverSampler


#DIR NAMES
DIR_NAME = "zipfilee_FILES/ds_data/"
TRAIN_FILE = "data_train.csv"
TEST_FILE = "data_test.csv"

#getting the train and test zipfilee_FILES
train = pd.read_csv(DIR_NAME+TRAIN_FILE)
test = pd.read_csv(DIR_NAME+TEST_FILE)

#preprocessing
#too many unique values as well as too many missing values, let's remove it
train.drop('num18', axis=1, inplace=True)
#too many missing values, and less categories, lets make the missing as another category, here 88,99 and 66 for example.
train.cat6.fillna(88, inplace=True)
train.cat8.fillna(99, inplace=True)
train.cat10.fillna(66, inplace=True)
#Replace these categories with the most frequent label.
agg_cat = ["cat1", "cat2", "cat3", "cat4", "cat5", "cat12"]
for i in agg_cat:
    max_ = train[i].value_counts()
    for j in max_.index:
        if max_[j] == max_.max():
            val = j
    train[i].fillna(j, inplace=True)
    print(i, " completed!")
#The rest less missing values
train.fillna(-999, inplace=True)

#---------------------------------------------------------------------------------------------------------------------------
#doing the same with test features
#too many unique values as well as too many missing values, let's remove it
test.drop('num18', axis=1, inplace=True)
#too many missing values, and less categories, lets make the missing as another category, here 88,99 and 66 for example.
test.cat6.fillna(88, inplace=True)
test.cat8.fillna(99, inplace=True)
test.cat10.fillna(66, inplace=True)
#Replace these categories with the most frequent label.
agg_cat = ["cat1", "cat2", "cat3", "cat4", "cat5", "cat12"]
for i in agg_cat:
    max_ = train[i].value_counts()
    for j in max_.index:
        if max_[j] == max_.max():
            val = j
    train[i].fillna(j, inplace=True)
    print(i, " completed!")
#The rest less missing values
train.fillna(-999, inplace=True)

col1 = ['num1', 'num2', 'num3', 'num4', 'num5', 'num6', 'num7', 'num8',
       'num9', 'num10', 'num11', 'num12', 'num13', 'num14', 'num15', 'num16',
       'num17', 'num19', 'num20', 'num21', 'num22', 'num23', 'der1', 'der2',
       'der3', 'der4', 'der5', 'der6', 'der7', 'der8', 'der9', 'der10',
       'der11', 'der12', 'der13', 'der14', 'der15', 'der16', 'der17', 'der18',
       'der19']
col2 = ['cat1', 'cat2', 'cat3', 'cat4', 'cat5', 'cat6', 'cat7', 'cat8',
       'cat9', 'cat10', 'cat11', 'cat12', 'cat13', 'cat14']

train[col1] = minmax_scale(train[col1])
test[col1] = minmax_scale(test[col1])

le = LabelEncoder()
for i in col2:
    train[i] = le.fit_transform(train[i])
    test[i] = le.fit_transform(test[i])

y = train.target
train.drop(["target", "id"], axis=1, inplace=True)
test.drop("id", axis=1, inplace=True)

#train test and validation split
x_train, x_test, y_train, y_test = train_test_split(train, y, test_size=0.1)
x_train, x_val, y_train, y_val = train_test_split(train, y, test_size=0.3)
print("TRAIN : ", x_train.shape , " and ", y_train.shape)
print("TEST : ", x_test.shape, " and ", y_test.shape)
print("VALIDATION : ", x_val.shape, " and ", y_val.shape)
print("MAIN TO PREDICT ", test.shape)

#Random Oversampling
ros = RandomOverSampler(random_state=0)
ros.fit(x_train, y_train)
X_resampledo, y_resampledo = ros.fit_sample(x_train, y_train)
print(X_resampledo.shape, y_resampledo.shape)

#model_selection
catboost_pool = Pool(X_resampledo, y_resampledo)
cat_model = CatBoostClassifier(task_type='CPU', iterations=20000, learning_rate=0.03, early_stopping_rounds=5)
cat_model.fit(X_resampledo, y_resampledo, verbose=True, plot=False, eval_set=(x_val, y_val),)

#accuracy on test categories
print(cat_model.score(x_test,y_test))

#metrics and score
y_pred = cat_model.predict(x_test)
print("ACCURACY SCORE : ", accuracy_score(y_test, y_pred))
print("MAE : ",mean_absolute_error(y_test, y_pred))
print("MSE : ", mean_squared_error(y_test, y_pred))
print("LOG LOSS : ", log_loss(y_test, y_pred))
print("COHEN KAPPA : ", cohen_kappa_score(y_test, y_pred))

#uncomment next lines to generate new csv results.
'''
y_proba = cat_model.predict_proba(test)
result = pd.DataFrame(data=y_proba, index=test.index)
result.to_csv("finalsub.csv")
'''

#------------------------------end--------------------------------------------------------