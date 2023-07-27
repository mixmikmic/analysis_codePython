import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

train=pd.read_csv('titanic_train.csv')

train.head()

train.info()

sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

sns.countplot(x='Survived',data=train)

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')

sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')

sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)

sns.countplot(x='SibSp',data=train)

train['Fare'].hist(bins=40,figsize=(8,4))

#Filling the missing data by Mean of columns by passenger class
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

train.drop('Cabin',axis=1,inplace=True)

train.head()

sex=pd.get_dummies(train['Sex'],drop_first=True)
embark=pd.get_dummies(train['Embarked'],drop_first=True)

train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

train=pd.concat([sex,embark,train],axis=1)

train.head()

train.columns

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)

#training and predicting
from sklearn.linear_model import LogisticRegression

#using logistic regression model to predict survival
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

predictions

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))

