"""
Created on Thu Jun 21 09:44:59 2018
@author: Silesh Chandran
"""

# Common imports
import numpy as np
import os
import pandas as pd
# to make this notebook's output stable across runs
np.random.seed(42)
# To plot pretty figures
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "end_to_end_project"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

def split_train_test(data, test_ratio):
    shuffled_indices= np.random_permutation(len(data))
    test_set_size= int(len(data) *test_ratio)
    test_indices =shuffled_indices[:test_set_size]
    train_indices =shuffled_indices[test_set_size:]
    return data.iloc[train_indices],data.iloc[test_indices]




housing= pd.read_csv("housing.csv")
housing["income_cat"]=np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"]<5,5.0,inplace=True)    

from sklearn.model_selection import StratifiedShuffleSplit

split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in split.split(housing,housing["income_cat"]):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]
    
housing["income_cat"].value_counts()/len(housing)*100

for s in (strat_train_set,strat_test_set):
    s.drop("income_cat",axis=1,inplace=True)    
    
housing=strat_train_set.copy()
housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.1,s=housing["population"]/100,label="population",figsize=(10,7),c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True)
plt.legend()
plt.show()

housing.corr()
housing["rooms_per_house"]=housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"]=housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

housing.corr()

from pandas.tools.plotting import scatter_matrix
attr= ["median_house_value","median_income","total_rooms","housing_median_age"]
scatter_matrix(housing[attr],figsize=(12,8))

housing= strat_train_set.drop("median_house_value",axis=1)
housing_labels=strat_train_set["median_house_value"].copy()

from sklearn.preprocessing import Imputer

imp= Imputer(strategy= "median")
housing_num =housing.drop("ocean_proximity",axis=1)
imp.fit(housing_num)
imp.statistics_
x=imp.transform(housing_num)
housing_tr= pd.DataFrame(x,columns=housing_num.columns)
housing_tr.info()

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat=housing["ocean_proximity"]
encoder.fit(housing_cat)
encoded_housing=encoder.fit_transform(housing_cat)
encoded_housing
encoder.classes_

from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder()
housing_cat_ohko=encoder.fit_transform(encoded_housing.reshape(-1,1))
housing_cat_ohko
housing_cat_ohko.toarray()

from sklearn.preprocessing import LabelBinarizer
encoder= LabelBinarizer()
housing_cat_ohko=encoder.fit_transform(housing_cat)

from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix,household_ix =3,4,5,6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self,add_bedrooms_per_room = True):
        self.add_bedrooms_per_room=add_bedrooms_per_room
    def fit(self, X,y=None):
        return self
    def transform(self,X,y=None):
        rooms_per_household=X[:,rooms_ix]/X[:,household_ix]
        population_per_household = X[:,population_ix]/X[:,household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room= X[:,bedrooms_ix]/X[:,rooms_ix]
            return np.c_[X,rooms_per_household,population_per_household,bedrooms_per_room]
        else:
            return np.c_[X,rooms_per_household,population_per_household]


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([('imputer',Imputer(strategy='median')),('attribs_adder',CombinedAttributesAdder()),('std_scaler',StandardScaler())])

housing_num_tr = num_pipeline.fit_transform(housing_num)

class DataFrameSelector(BaseEstimator,TransformerMixin):
    def __init__(self,attribute_names):
        self.attribute_names=attribute_names
    def fit(self,X,y=None):
        return self
    def transform(self,X):
        return X[self.attribute_names].values
    


num_attribs=list(housing_num)
cat_attribs=["ocean_proximity"]

num_pipeline = Pipeline([('selector',DataFrameSelector(num_attribs)),('imputer',Imputer(strategy='median')),('attribs_adder',CombinedAttributesAdder()),('std_scaler',StandardScaler()),])

class MyLabelBinarizer(TransformerMixin):
    def __init__(self, *args, **kwargs):
        self.encoder = LabelBinarizer(*args, **kwargs)
    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self
    def transform(self, x, y=0):
        return self.encoder.transform(x)

cat_pipeline = Pipeline([('selector',DataFrameSelector(cat_attribs)),('label_binarizer',MyLabelBinarizer()),])

from sklearn.pipeline import FeatureUnion

full_pipeline= FeatureUnion(transformer_list=[("num_pipeline",num_pipeline),("cat_pipeline",cat_pipeline),])

housing_prepared = full_pipeline.fit_transform(housing)

housing_prepared.shape

from sklearn.linear_model import LinearRegression

lin_reg= LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)

some_data = housing.iloc[:5]
some_labels= housing_labels.iloc[:5]
some_data_prep= full_pipeline.transform(some_data)

lin_reg.predict(some_data_prep)
some_labels


from sklearn.metrics import mean_squared_error
housing_predictions =lin_reg.predict(housing_prepared)
lin_mse=mean_squared_error(housing_labels,housing_predictions)
lin_rmse=np.sqrt(lin_mse)
lin_rmse

from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared,housing_labels)


housing_predictions =tree_reg.predict(housing_prepared)
lin_mse=mean_squared_error(housing_labels,housing_predictions)
lin_rmse=np.sqrt(lin_mse)
lin_rmse

from sklearn.model_selection import cross_val_score

scores= cross_val_score(tree_reg,housing_prepared,housing_labels,scoring="neg_mean_squared_error",cv=10)
tree_rmse_scores= np.sqrt(-scores)

def display_scores(scores):
    print("scores: ", scores)
    print("mean: ", scores.mean())
    print("standard deviation: ", scores.std())
    
display_scores(tree_rmse_scores)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
param_grid= [{'n_estimators':[3,10,30,50],'max_features':[2,4,6,8,10]},{'bootstrap':[False],'n_estimators':[3,10],'max_features':[2,3,4]},]
forest_reg=RandomForestRegressor()
grid_search = GridSearchCV(forest_reg,param_grid,cv=5,scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared,housing_labels)
grid_search.best_params_
grid_search.best_estimator_
cvres= grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"],cvres["params"]):
    print(np.sqrt(-mean_score), params)
    
feature_importances =grid_search.best_estimator_.feature_importances_
extra=["rooms_per_hhold","pop_per_hhold","bedrooms_per_room"]
cat_one_hot_attribs= list(encoder.classes_)
attributes= num_attribs+extra+cat_one_hot_attribs
sorted(zip(feature_importances,attributes),reverse=True)

final_model= grid_search.best_estimator_

x_test= strat_test_set.drop("median_house_value",axis=1)
y_test = strat_test_set["median_house_value"].copy()

x_test_prepared = full_pipeline.transform(x_test)

final_predictions= final_model.predict(x_test_prepared)

final_mse= mean_squared_error(y_test,final_predictions)
final_rmse= np.sqrt(final_mse)
final_rmse
