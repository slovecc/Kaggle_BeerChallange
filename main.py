import os
#change the working directory
working_directory = "/home/salvo/Desktop/SaturdayAI/challangeBeer/barcelona-beer-challenge/final_code/"
os.chdir(working_directory)
sys.path.append(working_directory)
#import local file
import cleaningData
import randomForest
import submission
import xgBoost
import catBoost
########## import library
import io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score

########################
#import data from the folder input
df2 =pd.read_csv(working_directory+'input/beer_train.csv')

#cleaning data
(new,imputer,label_encoder) =cleaningData.cleaning_data(df2)

#split for the models
X= pd.DataFrame(new).drop(['Style','Id'], axis = 1)
y=pd.DataFrame(new).loc[:,'Style']
print(X.head())
print(y.head())
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=13)

#models
#1st XG Boost
(model_xg,final_max_xg)=xgBoost.xg_boost(X_train, X_test, y_train, y_test)

#2nd cat Boost
(model_cat,final_max_cat)=catBoost.cat_boost(X_train, X_test, y_train, y_test)

###put together the two prediction and check how many different cases
pred_final = pd.concat([final_max_xg, final_max_cat], axis=1)
pred_final.columns = ['probXG', 'XG','probCAT', 'CAT']

cont=1
for index,row in pred_final.iterrows():
    if row['XG'] != row['CAT']:
        cont =cont+1
print(cont)

#submission
#read test file and apply the same preprocessing procedure
test = pd.read_csv('input/beer_test.csv')
submission.submission_file(test,model_xg,model_cat,imputer,label_encoder)