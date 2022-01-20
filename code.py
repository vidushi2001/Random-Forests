# importing required libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing dataset

train_df = pd.read_csv('/content/train.csv')
train_df.head()
train_df.info()
test_df = pd.read_csv('/content/test.csv')

# data pre-processing
train_df.drop(columns='Name',inplace=True)
a = train_df.loc[train_df['Survived'] > 0]
train_df['Age'].fillna(a['Age'].median(),inplace = True)
train_df.drop(columns = 'Cabin',inplace = True)
train_df.drop(columns = 'Embarked',inplace=True)
test_df.drop(['Name','Cabin','Ticket'],inplace=True,axis='columns')
test_df.drop(columns='Embarked',inplace=True)
test_df['Age'].fillna(test_df['Age'].median(),inplace = True)
test_df.dropna(inplace=True)

# data transformation
np.unique(train_df['Sex'])
x = x.replace({"male":0,"female":1})
test_df.replace({"male":0,"female":1},inplace=True)

x = train_df.iloc[:,2:]
y = train_df.iloc[:,1:2]

# fitting the model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10,criterion = "entropy")
x.drop(columns='Ticket',inplace=True)
classifier.fit(x,y)

y_pred = classifier.predict(test_df.iloc[:,1:])
actual = pd.read_csv('/content/gender_submission.csv')
test_df = pd.merge(test_df,act,on='PassengerId')
acts = test_df.iloc[:,7]

# evaluating the model

from sklearn.metrics import confusion_matrix,accuracy_score,precision_score
cm = confusion_matrix(acts,y_pred)
accuracy = accuracy_score(acts,y_pred)
precision = precision_score(acts,y_pred)
