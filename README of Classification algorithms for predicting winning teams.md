import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing
%matplotlib inline

# Loading dataset
df = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0120ENv3/Dataset/ML0101EN_EDX_skill_up/cbb.csv')
df.head()

# Next we'll add a column that will contain "true" if the wins above bubble are over 7 and "false" if not. We'll call this column Win Index or "windex" for short.
df['windex'] = np.where(df.WAB > 7, 'True', 'False')

# we'll filter the data set to the teams that made the Sweet Sixteen, the Elite Eight, and the Final Four in the post season. We'll also create a new dataframe that will hold the values with the new column.
df1 = df[df['POSTSEASON'].str.contains('F4|S16|E8', na=False)]
df1.head()

# Convert Categorical features to numerical values
df1.groupby(['windex'])['POSTSEASON'].value_counts(normalize=True)
# 12% of teams with 6 or less wins above bubble make it into the final four while 18% of teams with 7 or more do. Lets convert wins above bubble (winindex) under 7 to 0 and over 7 to 1:

df['windex'].replace(to_replace=['False','True'], value=[0,1],inplace=True)
df.head()

# Use one hot encoding technique to conver categorical varables to binary variables and append them to the feature Data Frame
Feature = df1[['ADJOE','ADJDE','BARTHAG','EFG_O','EFG_D']]
Feature = pd.concat([Feature,pd.get_dummies(df1['POSTSEASON'])], axis=1)
Feature.drop(['S16'], axis = 1,inplace=True)
Feature.head()

# Lets defind feature sets, X:
X = Feature

# normalize data
X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]

# Training and Validation
# We split the X into train and test to find the best k
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Validation set:', X_val.shape,  y_val.shape)


## K Nearest Neighbor(KNN)
# Question 1 Build a KNN model using a value of k equals three, find the accuracy on the validation data (X_val and y_val)

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
k= 3
kneigh= KNeighborsClassifier(n_neighbors= 3).fit(X_train, y_train)
kneigh
yhatk= kneigh.predict(X_val)
print('the accuracy of the model is:', accuracy_score(y_val, yhatk))

# Question  2 Determine the accuracy for the first 15 values of k the on the validation data:
k= 15
mean_accuracy= np.zeros(k)
std_accuracy= np.zeros(k)
confusion_matrix= []

for n in range(1,k):
    neigh= KNeighborsClassifier(n_neighbors= n).fit(X_train, y_train)
    yhatk= neigh.predict(X_val)
    mean_accuracy[n]= accuracy_score(y_val, yhatk)
    std_accuracy[n]= np.std(yhatk== y_val)/ np.sqrt(yhatk.shape[0])
    
mean_accuracy[0:16]

## Decision Tree
from sklearn.tree import DecisionTreeClassifier
basketballtree= DecisionTreeClassifier(criterion= 'entropy', max_depth= 4)
basketballtree.fit(X_train, y_train)
predtree= basketballtree.predict(X_val)
print('the accuracy score is:', accuracy_score(y_val, predtree))

# Question  3 Determine the minumum   value for the parameter max_depth that improves results
from matplotlib import pyplot as plt
max_depth_test= np.linspace(1, 20, 20, endpoint=True)
accuracy_testdata= []
accuracy_traindata= []

for n in max_depth_test:
    basketballtree1= DecisionTreeClassifier(criterion= 'entropy', max_depth= n)
    basketballtree1.fit(X_train, y_train)
    
    predtree1= basketballtree1.predict(X_val)
    pred_train_data= basketballtree1.predict(X_train)
    
    
    accuracy_testdata.append(accuracy_score(y_val, predtree1))
    accuracy_traindata.append(accuracy_score(y_train, pred_train_data))

    
print(accuracy_testdata)
print(accuracy_traindata)


line1= plt.plot(max_depth_test, accuracy_testdata, 'b', label= 'test data points')
line2= plt.plot(max_depth_test, accuracy_traindata, 'r', label= 'train data points')
plt.xlabel('max depths')
plt.ylabel('accuracy score')
plt.show()

# minimum value of max_depth would be 2 that improves the results from the lower value

## Support Vector Machine
# Question  4Train the following linear  support  vector machine model and determine the accuracy on the validation data
from sklearn import svm
Var_svm= svm.SVC(kernel= 'linear')
Var_svm.fit(X_train, y_train)
vektor_predict= Var_svm.predict(X_val)
vektor_predict

from sklearn.metrics import jaccard_similarity_score
print('The accuracy score is:', jaccard_similarity_score(y_val, vektor_predict))

## Logistic Regression
# Train a logistic regression model and determine the accuracy of the validation data (set C=0.01)
from sklearn.linear_model import LogisticRegression
LR= LogisticRegression(C= 0.01, solver= 'liblinear')
LR.fit(X_train, y_train)
yhat_LR= LR.predict(X_val)

# calculating probability
yhat2_LR= LR.predict_proba(X_val)

from sklearn.metrics import jaccard_similarity_score
print('Accuracy of the model is:', jaccard_similarity_score(y_val, yhat_LR))

# OR
from sklearn.metrics import log_loss
print('The model probability accuracy is:', log_loss(y_val, yhat2_LR))

## Model Evaluation using Test set
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

def jaccard_index(predictions, true):
    if (len(predictions) == len(true)):
        intersect = 0;
        for x,y in zip(predictions, true):
            if (x == y):
                intersect += 1
        return intersect / (len(predictions) + len(true) - intersect)
    else:
        return -1
        
 # Question  5 Calculate the  F1 score and Jaccard Similarity score for each model from above. Use the Hyperparameter that performed best on the validation data.

test_df = pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0120ENv3/Dataset/ML0101EN_EDX_skill_up/basketball_train.csv',error_bad_lines=False)
test_df.head()

test_df['windex'] = np.where(test_df.WAB > 7, 'True', 'False')
test_df1 = test_df[test_df['POSTSEASON'].str.contains('F4|S16|E8', na=False)]
test_df1. head()
test_df1.groupby(['windex'])['POSTSEASON'].value_counts(normalize=True)
test_Feature = test_df1[['ADJOE','ADJDE','BARTHAG','EFG_O','EFG_D']]
test_Feature = pd.concat([test_Feature,pd.get_dummies(test_df1['POSTSEASON'])], axis=1)
test_Feature.drop(['S16'], axis = 1,inplace=True)
test_Feature.head()
test_X=test_Feature
test_X= preprocessing.StandardScaler().fit(test_X).transform(test_X)
test_X[0:5]

## KNN

KNN= KNeighborsClassifier(n_neighbors= 3).fit(X_train, y_train).predict(test_X)
print('The f1 score is:', f1_score(test_y, KNN, average= 'weighted'))
print('Jaccard similarity score for this model is:', jaccard_similarity_score(test_y, KNN))

## Decision Tree
DT= DecisionTreeClassifier(criterion= 'entropy', max_depth= 2).fit(X_train, y_train).predict(test_X) #i changed the max_depth according to last analysis
print('The f1 score for decision tree is:', f1_score(test_y, DT, average= 'weighted'))
print('Jaccard similarity score for this model is:', jaccard_similarity_score(test_y, DT))

## Support Vektor Machine
SVM_model= svm.SVC(kernel= 'linear').fit(X_train, y_train).predict(test_X)
print('The f1 score for SVM is:', f1_score(y_val, vektor_predict, average= 'weighted'))
print('Jaccard similarity score for this model is:', jaccard_similarity_score(test_y, SVM_model))

## Logistic Regression

LR_model= LR= LogisticRegression(C= 0.01, solver= 'liblinear').fit(X_train, y_train).predict(test_X)
print('The f1 score for logistic regression is:', f1_score(test_y, LR_model, average= 'weighted'))
print('Jaccard similarity score for this model for this model is:', jaccard_similarity_score(test_y, LR_model))





