#!/usr/bin/python

""" Created on February 2020
@author: Sara Alizadeh """

import sys
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from Helper.ML_Helper import MlHelper

def predict_age(row_age_pclass):
    age = row_age_pclass[0]
    pclass = row_age_pclass[1]
    if pd.isnull(age):
        if pclass == 1:
            return 38
        elif pclass == 2:
            return 29
        else:
            return 25
    else:
        return age


if __name__ == "__main__":
    try:
        train_df = pd.read_csv("train.csv")
        test_df = pd.read_csv("test.csv")
        #print(train_df.isnull().sum())
        #train_df.info()
        #print(train_df.corr().abs())
        train_df.drop(['PassengerId', 'Name', 'Cabin'], axis = 1, inplace=True)
        train_df.Embarked.fillna(method='ffill', inplace=True)
        train_df['Age'] = train_df[['Age', 'Pclass']].apply(predict_age, axis=1) 
        #print(train_df.isnull().sum())
        le = LabelEncoder()
        object_cols_train = train_df.select_dtypes('object').columns
        train_df[object_cols_train] = train_df[object_cols_train].apply(le.fit_transform) 
        #train_df.info()
        #print(train_df.corr().Survived.abs().sort_values())

        #clean test data
        test_df.drop(['Name', 'Cabin'], axis = 1, inplace=True)
        test_df.Fare.fillna(method='ffill', inplace=True)
        test_df['Age'] = test_df[['Age', 'Pclass']].apply(predict_age, axis=1) 
        #print(test_df.isnull().sum())
        le = LabelEncoder()
        object_cols_test = test_df.select_dtypes('object').columns
        test_df[object_cols_test] = test_df[object_cols_test].apply(le.fit_transform) 

        X = train_df.drop('Survived', axis=1).values
        y = train_df.Survived.values
        X_test_final = test_df.drop('PassengerId', axis=1).values

        #scale
        scale = StandardScaler()
        scale.fit(X)
        X = scale.transform(X)
        scale_test = StandardScaler()
        scale_test.fit(X_test_final)
        X_test_final = scale_test.transform(X_test_final)

        #Linear Regression 77.033%
        ml_model = LogisticRegression()
        ml_model = MlHelper(ml_model, X, y, X_test_final)
        ml_model.train_and_evaluate_with_splitted_data()
        ml_model.train_and_evaluate_with_cross_validation()

        ml_model_test = LogisticRegression()
        ml_model_test = MlHelper(ml_model_test, X, y, X_test_final)
        y_test_final = ml_model_test.train_and_predict_with_whole_data()
        output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': y_test_final})
        output.to_csv('Titanic_LogReg.csv', index=False)
        print("CSV was successfully saved!")
        
        #Random Forest 77.511%
        ml_model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=1)
        ml_model = MlHelper(ml_model, X, y, X_test_final)
        ml_model.train_and_evaluate_with_splitted_data()
        ml_model.train_and_evaluate_with_cross_validation()

        ml_model_test = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=1)
        ml_model_test = MlHelper(ml_model_test, X, y, X_test_final)
        y_test_final = ml_model_test.train_and_predict_with_whole_data()
        output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': y_test_final})
        output.to_csv('Titanic_RanFor.csv', index=False)
        print("CSV was successfully saved!")
        
        #SVC 77.990%
        ml_model = SVC()
        ml_model = MlHelper(ml_model, X, y, X_test_final)
        ml_model.train_and_evaluate_with_splitted_data()
        ml_model.train_and_evaluate_with_cross_validation()

        ml_model_test = SVC()
        ml_model_test = MlHelper(ml_model_test, X, y, X_test_final)
        y_test_final = ml_model_test.train_and_predict_with_whole_data()
        output = pd.DataFrame({'PassengerId': test_df.PassengerId, 'Survived': y_test_final})
        output.to_csv('Titanic_SVC.csv', index=False)
        print("CSV was successfully saved!")


      





    except:
        pass

