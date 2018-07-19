# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 18:04:05 2018

@author: Silesh Chandran
"""

import numpy as np
import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder    
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import LabelBinarizer


#to return reshaped array after labelencoding
class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)
    
#to select required data (categorical and numerical)    
class DataFrameSelector(BaseEstimator,TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names=attribute_names
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        return X[self.attribute_names].values    

#function to display cross validation scores 
def display_scores(scores):
    print("scores: ", scores)
    print("mean: ", scores.mean())
    print("standard deviation: ", scores.std())




#opening csv file
housing= pd.read_csv("housing.csv")

#creating income as categorical variable for stratified spliting
housing["income_cat"]=np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"]<5,5.0,inplace=True)   

#splitting into test and train sets
split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing["income_cat"]):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index] 
    
#removing income_cat 
for s in (strat_train_set,strat_test_set):
    s.drop("income_cat",axis=1,inplace=True) 
housing=strat_train_set.copy()

#correleation matrix and scatter plot
housing.corr()   
#from pandas.tools.plotting import scatter_matrix
#attr= ["median_house_value","median_income","total_rooms","housing_median_age"]
#scatter_matrix(housing[attr],figsize=(12,8))

#X and Y(input and labels)
housing= strat_train_set.drop("median_house_value",axis=1)
housing_labels=strat_train_set["median_house_value"].copy()

#adding 3 derived attributes
housing["rooms_per_house"]=housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"]=housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

#numerical and categorical data separated
housing_num =housing.drop("ocean_proximity",axis=1)
num_attribs=list(housing_num)
housing_cat=housing["ocean_proximity"]
cat_attribs=["ocean_proximity"]

#full numerical pipeline
num_pipeline = Pipeline([('selector',DataFrameSelector(num_attribs)),('imputer',Imputer(strategy='median')),('std_scaler',StandardScaler())])
#full categorical pipeline
cat_pipeline = Pipeline([('selector',DataFrameSelector(cat_attribs)),('label_encoder',MyLabelBinarizer())])
#combining both
full_pipeline= FeatureUnion(transformer_list=[("num_pipeline",num_pipeline),("cat_pipeline",cat_pipeline),])
housing_prepared = full_pipeline.fit_transform(housing)

#linear regression
lin_reg= LinearRegression()

#decision tree regression
tree_reg = DecisionTreeRegressor()

#random forest regression
forest_reg=RandomForestRegressor()

#Support vector machine regression
svm_reg= SVR()

#performing training and cross validation on all three
lregscores= cross_val_score(lin_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error",cv=10)
lin_reg_scores= np.sqrt(-lregscores)
dtreescores= cross_val_score(tree_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error",cv=10)
tree_rmse_scores= np.sqrt(-dtreescores)
forestscores= cross_val_score(forest_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error",cv=10)
forest_reg_scores= np.sqrt(-forestscores)
svmscores= cross_val_score(svm_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error",cv=10)
svm_reg_scores= np.sqrt(-svmscores)

#displaying 3 scores
display_scores(tree_rmse_scores)
display_scores(lin_reg_scores)
display_scores(forest_reg_scores)
display_scores(svm_reg_scores)

#proceding with random forest regression and doing grid search to find best parameters
param_grid= [{'n_estimators':[3,10,30],'max_features':[2,4,6,8]},{'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]},]
grid_search = GridSearchCV(forest_reg,param_grid,cv=5,scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared,housing_labels)
grid_search.best_params_
model=grid_search.best_estimator_

#grid search
feature_importances =grid_search.best_estimator_.feature_importances_
#defining attribute names
encoder = LabelEncoder()
encoded_housing=encoder.fit_transform(housing_cat)
cat_one_hot_attribs=list(encoder.classes_)
attributes= num_attribs+cat_one_hot_attribs
print(sorted(zip(feature_importances,attributes),reverse=True))

#new transformer to select the top attributes only
def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

indices_of_top_k(feature_importances, 10)
class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]
 
#complete pipeline that does selection of top 10 best features and uses the best estimator
mod_full_pipeline = Pipeline([('full_pipeline',full_pipeline()),('top_selector',TopFeatureSelector(feature_importances,10),('model',model()))])

#testing and finishing
x_test= strat_test_set.drop("median_house_value",axis=1)
y_test = strat_test_set["median_house_value"].copy()

final= mod_full_pipeline.fit(x_test)

final_mse= mean_squared_error(y_test,mod_full_pipeline.predict(x_test))
final_rmse= np.sqrt(final_mse)
final_rmse


