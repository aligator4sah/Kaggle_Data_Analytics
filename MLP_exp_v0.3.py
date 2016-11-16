
# coding: utf-8

# In[1]:

#MLP trial
#version:0.3

import pandas #provides data structures to quickly analyze data
#Since this code runs on Kaggle server, data can be accessed directly in the 'input' folder
#Read the train dataset
dataset = pandas.read_csv("/home/sabrina/Downloads/train.csv") 

#Read test dataset
dataset_test = pandas.read_csv("/home/sabrina/Downloads/test.csv")
#Save the id's for submission file
ID = dataset_test['id']
#Drop unnecessary columns
dataset_test.drop('id',axis=1,inplace=True)

#Print all rows and columns. Dont hide any
pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)

# Size of the dataframe

print(dataset.shape)

dataset = dataset.iloc[:,1:]


# In[2]:

import pandas
#Variable to hold the list of variables for an attribute in the train and test data
labels = []
#range of features considered
split=116
#number of features considered
size = 15

#create a dataframe with only continuous features
data=dataset.iloc[:,split:] 

#get the names of all the columns
cols=data.columns
cols = dataset.columns

for i in range(0,split):
    train = dataset[cols[i]].unique()
    test = dataset_test[cols[i]].unique()
    labels.append(list(set(train) | set(test)))    

del dataset_test


# In[4]:

#Import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

#One hot encode all categorical attributes
cats = []
for i in range(0, split):
    #Label encode
    label_encoder = LabelEncoder()
    label_encoder.fit(labels[i])
    feature = label_encoder.transform(dataset.iloc[:,i])
    feature = feature.reshape(dataset.shape[0], 1)

    #One hot encode
    #onehot_encoder = OneHotEncoder(sparse=False,n_values=len(labels[i]))
    #feature = onehot_encoder.fit_transform(feature)
    cats.append(feature)

import numpy
# Make a 2D array from a list of 1D arrays
encoded_cats = numpy.column_stack(cats)

# Print the shape of the encoded data
print(encoded_cats.shape)


# In[5]:

#Concatenate encoded attributes with continuous attributes
dataset_encoded = numpy.concatenate((encoded_cats,dataset.iloc[:,split:].values),axis=1)
del cats
del feature
del dataset
del encoded_cats
print(dataset_encoded.shape)


# In[7]:

#split train data to train and valiadation
#get the number of rows and columns
r, c = dataset_encoded.shape
#create an array which has indexes of columns
i_cols = []
for i in range(0,c-1):
    i_cols.append(i)

#Y is the target column, X has the rest
X = dataset_encoded[:,0:(c-1)]
Y = dataset_encoded[:,(c-1)]
del dataset_encoded


# In[8]:

#Validation chunk size
val_size = 0.1

#Use a common seed in all experiments so that same chunk is used for validation
seed = 0

#Split the data into chunks
from sklearn import cross_validation
X_train, X_val, Y_train, Y_val = cross_validation.train_test_split(X, Y, test_size=val_size, random_state=seed)
del X
del Y


# In[6]:

#All features
X_all = []

#List of combinations
comb = []

#Dictionary to store the MAE for all algorithms 
mae = []

#Scoring parameter1
from sklearn.metrics import mean_absolute_error

#Add this version of X to the list 
n = "All"
#X_all.append([n, X_train,X_val,i_cols])
X_all.append([n, i_cols])


# In[9]:

#Evaluation of various combinations of multi-layer perceptrons

#Import libraries for deep learning
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense


# In[10]:

# define baseline model
def baseline(v):
     # create model
     model = Sequential()
     model.add(Dense(v*(c-1), input_dim=v*(c-1), init='normal', activation='relu'))
     model.add(Dense(1, init='normal'))
     # Compile model
     model.compile(loss='mean_absolute_error', optimizer='adam')


# In[11]:

# define smaller model
def smaller(v):
     # create model
     model = Sequential()
     model.add(Dense(v*(c-1)/2, input_dim=v*(c-1), init='normal', activation='relu'))
     model.add(Dense(1, init='normal', activation='relu'))
     # Compile model
     model.compile(loss='mean_absolute_error', optimizer='adam')
     return model


# In[12]:

# define deeper model
def deeper(v):
     # create model
     model = Sequential()
     model.add(Dense(v*(c-1), input_dim=v*(c-1), init='normal', activation='relu'))
     model.add(Dense(v*(c-1)/2, init='normal', activation='relu'))
     model.add(Dense(1, init='normal', activation='relu'))
     # Compile model
     model.compile(loss='mean_absolute_error', optimizer='adam')
     return model


# In[13]:

# Optimize using dropout and decay
from keras.optimizers import SGD
from keras.layers import Dropout
from keras.constraints import maxnorm


# In[14]:

def dropout(v):
    #create model
    model = Sequential()
    model.add(Dense(v*(c-1), input_dim=v*(c-1), init='normal', activation='relu',W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(v*(c-1)/2, init='normal', activation='relu', W_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.add(Dense(1, init='normal', activation='relu'))
    # Compile model
    sgd = SGD(lr=0.1,momentum=0.9,decay=0.0,nesterov=False)
    model.compile(loss='mean_absolute_error', optimizer=sgd)
    return model


# In[15]:

# define decay model
def decay(v):
    # create model
    model = Sequential()
    model.add(Dense(v*(c-1), input_dim=v*(c-1), init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='relu'))
    # Compile model
    sgd = SGD(lr=0.1,momentum=0.8,decay=0.01,nesterov=False)
    model.compile(loss='mean_absolute_error', optimizer=sgd)
    return model


# In[16]:

est_list = []
#uncomment the below if you want to run the algo
est_list = [('MLP',baseline),('smaller',smaller),('deeper',deeper),('dropout',dropout),('decay',decay)]


# In[17]:

for name, est in est_list:
 
    algo = name

    #Accuracy of the model using all features
    for m,i_cols_list in X_all:
        model = KerasRegressor(build_fn=est, v=1, nb_epoch=10, verbose=0)
        model.fit(X_train[:,i_cols_list],Y_train)
        result = mean_absolute_error(numpy.expm1(Y_val), numpy.expm1(model.predict(X_val[:,i_cols_list])))
        mae.append(result)
        print(name + " %s" % result)
        
    comb.append(algo )


# In[ ]:



