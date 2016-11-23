
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline
from sklearn.preprocessing import LabelEncoder

train_data = pd.read_csv("F:/GraduateStudy/2016Fall/DatAna/TermProject/train.csv")
test_data = pd.read_csv("F:/GraduateStudy/2016Fall/DatAna/TermProject/test.csv")

#print("Train data dimensions: ", train_data.shape)
#print("Test data dimensions: ", test_data.shape)
train_data = train_data.iloc[:,1:]

train_data.info()
test_data.info()


# In[2]:

#Updating loss
train_data['loss'] = np.log1p(train_data["loss"])

# sepearte the categorical and continous features
cont_columns = []
cat_columns = []

for i in train_data.columns:
    if train_data[i].dtype == 'float':
        cont_columns.append(i)
    elif train_data[i].dtype == 'object':
        cat_columns.append(i)

for cf1 in cat_columns:
    le = LabelEncoder()
    le.fit(train_data[cf1].unique())
    train_data[cf1] = le.transform(train_data[cf1])

train_data.head(20)


# In[3]:

ntrain = train_data.shape[0]
ntest = test_data.shape[0]

features = [x for x in train_data.columns if x not in ['id','loss']]

train_test = pd.concat((train_data[features], test_data[features])).reset_index(drop=True)

#train_test = np.concatenate((train_data_pred[features],test_data[features]),axis=0)

for f in train_test.columns: 
    if train_test[f].dtype=='object': 
        lbl = LabelEncoder() 
        lbl.fit(list(train_test[f].values)) 
        train_test[f] = lbl.transform(list(train_test[f].values))


train_x = train_test.iloc[:ntrain,:]

test_x = train_test.iloc[ntrain:,:]
    
train_x = np.array(train_x);
test_x = np.array(test_x);
print(test_x)


# In[8]:

#get the number of rows and columns
r, c = train_x.shape

#Y is the target column, X has the rest
X = train_x
Y = train_data['loss']

#print X.shape
print ntrain

#Validation chunk size
val_size = 0.4

#Use a common seed in all experiments so that same chunk is used for validation
seed = 0

from sklearn import cross_validation
X_train, X_val, Y_train, Y_val = cross_validation.train_test_split(X, Y, test_size=val_size, random_state=seed)

del X
del Y

i_cols = []
for i in range(0,c-1):
    i_cols.append(i)

#All features
X_all = []

#List of combinations
comb = []

#Dictionary to store the MAE for all algorithms 
mae = []

#Scoring parameter
from sklearn.metrics import mean_absolute_error

#Add this version of X to the list 
n = "All"
#X_all.append([n, X_train,X_val,i_cols])
X_all.append([n, i_cols])


# In[9]:

#Evaluation of various combinations of LinearRegression

#Import the library
from sklearn.linear_model import LinearRegression

##Set the base model
model = LinearRegression(n_jobs=-1)
algo = "LR"
#
##Accuracy of the model using all features
for name,i_cols_list in X_all:
    model.fit(X_train[:,i_cols_list],Y_train)
    result = mean_absolute_error(numpy.expm1(Y_val), numpy.expm1(model.predict(X_val[:,i_cols_list])))
    mae.append(result)
    print(name + " %s" % result)
comb.append(algo)


# In[ ]:



