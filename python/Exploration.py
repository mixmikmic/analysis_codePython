get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


from itertools import product

sns.set_style('whitegrid')

import warnings
warnings.filterwarnings('ignore')


SEED = 2313
np.random.seed(SEED)

# load data
train = pd.read_csv('../data/train.csv')
test  = pd.read_csv('../data/test.csv')
sub   = pd.read_csv('../data/Sample_Submission.csv')

# All of the users in the test set are present in the training set as well.
test.User_ID.isin(train.User_ID).value_counts()

# Only 61 products are not present in the training dataset.
test.Product_ID.isin(train.Product_ID).value_counts()

sns.kdeplot(train.Purchase);

train.groupby('Age')['Purchase'].mean().plot(kind='bar')
plt.ylabel('Mean Purchase')
plt.title('Relationship between Age and Mean Purchase');

train.columns

sns.countplot(x='Occupation', data=train);

sns.pointplot(x='Occupation', y='Purchase', data=train);

sns.pointplot(x="Occupation", y="Purchase", hue="Age", data=train, estimator=np.mean);

train.loc[(train.Age == '55+') & (train.Occupation == 5), :]

sns.pointplot(x="Gender", y="Purchase", hue="Stay_In_Current_City_Years", data=train, estimator=np.mean);

sns.pointplot(x="Gender", y="Purchase", hue="Marital_Status", data=train, estimator=np.mean);

sns.pointplot(x="Gender", y="Purchase", hue="Age", data=train, estimator=np.mean);

sns.pointplot(x="Gender", y="Purchase", hue="Age", data=train, estimator=np.median);

sns.countplot(x='Marital_Status', data=train);

sns.countplot(x="Gender", data=train);

sns.countplot(x='Age', data=train);

# To capture relationship between different users and products that they get attracted to
# we can plot pivot table to see if there is any relationship between age segments and purchase amount.

ss = pd.pivot_table(train, index=['Gender', 'Marital_Status', 'Age'], columns='Product_Category_1', values='Purchase', fill_value=0)

def plot_diff_purchase_pattern(pivot_df, key1, key2, label):
    """
    Pivot Table with index as (Gender, Marital Status and Age)
    and Column is Product_Category_1 with Purchase as the value.
    
    So for every product category we can see the change in purchase ability
    based on the keys passed.
    """
    
    (ss.ix[key1] - ss.ix[key2]).plot(label=label)

train.Age.unique()

plot_diff_purchase_pattern(ss, ('F', 0, '0-17'), ('M', 0, '0-17'), '0-17')
plot_diff_purchase_pattern(ss, ('F', 0, '18-25'), ('M', 0, '18-25'), '18-25')
plot_diff_purchase_pattern(ss, ('F', 0, '36-45'), ('M', 0, '36-45'), '36-45')
plot_diff_purchase_pattern(ss, ('F', 0, '46-50'), ('M', 0, '46-50'), '46-50')
plot_diff_purchase_pattern(ss, ('F', 0, '51-55'), ('M', 0, '51-55'), '51-55')
plot_diff_purchase_pattern(ss, ('F', 0, '55+'), ('M', 0, '55+'), '55+')
plt.xticks(np.arange(1, 21))
plt.legend(loc='best');

plot_diff_purchase_pattern(ss, ('F', 0, '18-25'), ('F', 1, '18-25'), '18-25')
plot_diff_purchase_pattern(ss, ('F', 0, '26-35'), ('F', 1, '26-35'), '26-35')
plot_diff_purchase_pattern(ss, ('F', 0, '36-45'), ('F', 1, '36-45'), '36-45')
plot_diff_purchase_pattern(ss, ('F', 0, '46-50'), ('F', 1, '46-50'), '46-50')
plot_diff_purchase_pattern(ss, ('F', 0, '51-55'), ('F', 1, '51-55'), '51-55')
plot_diff_purchase_pattern(ss, ('F', 0, '55+'), ('F', 1, '55+'), '55+')
plt.xticks(np.arange(1, 21))
plt.legend(loc='best');

plot_diff_purchase_pattern(ss, ('M', 0, '18-25'), ('M', 1, '18-25'), '18-25')
plot_diff_purchase_pattern(ss, ('M', 0, '26-35'), ('M', 1, '26-35'), '26-35')
plot_diff_purchase_pattern(ss, ('M', 0, '36-45'), ('M', 1, '36-45'), '36-45')
plot_diff_purchase_pattern(ss, ('M', 0, '46-50'), ('M', 1, '46-50'), '46-50')
plot_diff_purchase_pattern(ss, ('M', 0, '51-55'), ('M', 1, '51-55'), '51-55')
plot_diff_purchase_pattern(ss, ('M', 0, '55+'), ('M', 1, '55+'), '55+')
plt.xticks(np.arange(1, 21))
plt.legend(loc='best');



