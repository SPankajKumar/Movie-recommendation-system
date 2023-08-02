import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from pandas import read_csv
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, explained_variance_score, r2_score
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

path= r"movie_metadata.csv"

reader=pd.read_csv('movie_metadata.csv')

global minval
global maxval
global minmax
global category_features
global number_features
minmax = preprocessing.MinMaxScaler()

reader.drop(['movie_imdb_link'], axis=1,inplace=True)

text_features = []
category_features = ['genres','movie_title','color','director_name','actor_2_name','actor_1_name','actor_3_name','language','country','content_rating','plot_keywords']
number_features = ['num_critic_for_reviews','duration','director_facebook_likes','actor_3_facebook_likes','actor_1_facebook_likes','gross','num_voted_users','cast_total_facebook_likes','facenumber_in_poster','num_user_for_reviews','budget','title_year','actor_2_facebook_likes','imdb_score','aspect_ratio','movie_facebook_likes']
allfeatures=list(reader.columns)
#allfeatures = ['actor_1_name', 'actor_2_name', 'actor_3_name', 'director_name', 'country', 'content_rating', 'language', 'actor_1_facebook_likes', 'actor_2_facebook_likes', 'actor_3_facebook_likes', 'director_facebook_likes','cast_total_facebook_likes','budget', 'gross', 'genres', "imdb_score"]
imp_non_elim=['actor_1_name', 'actor_2_name','actor_3_name','director_name','country','language','gross','imdb_score','budget','genres','duration']

elim=[]
for i in allfeatures:
  if i not in imp_non_elim:
    elim+=[i]
print(elim)    

reader.head(5)

reader.isna().sum()

def cleaned_data(dupereader):
    global text_features
    global category_features
    global numerical_features        
    selected_data = dupereader
    data = selected_data.dropna(axis = 0, how = 'any',subset=elim)#add subset list later, list is currently wrong, take inverse of list
    data = data.reset_index(drop = True)
    for x in category_features:
        data[x] = data[x].fillna('None').astype('category')
    for y in number_features:
        data[y] = data[y].fillna(0.0).astype(np.float)
    return data

def pre_categorical(data):
    label_encoder = LabelEncoder()
    label_encoded_data = label_encoder.fit_transform(data) 
    return label_encoded_data

reader1=reader
data =cleaned_data(reader1)


label_encoder = LabelEncoder()
for i in category_features:
  data[i]= label_encoder.fit_transform(data[i])



data.head(5)

for i in number_features:
  x = np.array(list(data[i]))
  x = x.reshape(-1, 1)
  x_scaled = minmax.fit_transform(x)
  x = x_scaled.reshape(1,len(x))
  data[i] = x[0]

for i in category_features:
  x = np.array(list(data[i]))
  x = x.reshape(-1, 1)
  x_scaled = minmax.fit_transform(x)
  x = x_scaled.reshape(1,len(x))
  data[i] = x[0]

#for i in plot_keywords:
#  data[i]=data[i].apply(str.split("|"))
#print(data.plot_keywords)

#vectorizer = TfidfVectorizer()
#for i in text_features:
#  x = list(data[i])
#  a=vectorizer.vocabulary
#  print(a)
#  vector = vectorizer.fit_transform(x)
#  x= vector.toarray()
#  data[i] = x[0]

print(len(data.index))

pd.plotting.scatter_matrix(data)

from sklearn import linear_model

data2=data
training_data=data2.head(3533)

test_data = data2.iloc[3534:]

independent=allfeatures
independent.remove("gross")

dependent=training_data[["gross"]]

independent=training_data.loc[:, data.columns != 'name']


regr = linear_model.LinearRegression()
regr.fit(independent, dependent)

r2 = regr.score(independent, dependent)
print(r2)

import statsmodels.api as sm
independent1 = sm.add_constant(independent)
result = sm.OLS(dependent, independent1).fit()
#print dir(result)
print(result.rsquared, result.rsquared_adj)