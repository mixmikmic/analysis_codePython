get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np
import os, sys
import re

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_validation import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer, MinMaxScaler
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import TruncatedSVD

from scipy import stats

from itertools import combinations

import warnings
warnings.filterwarnings('ignore')

np.random.seed(1)

basepath = os.path.expanduser('~/Desktop/src/Loan_Default_Prediction/')
sys.path.append(os.path.join(basepath, 'src'))

from data import *

def get_stratified_sample(X, y, train_size, random_state=10):
    """
    Takes in a feature set and target with percentage of training size and a seed for reproducability.
    Returns indices for the training and test sets.
    """
    
    itrain, itest = train_test_split(range(len(X)), stratify=y, train_size=train_size, random_state=random_state)
    return itrain, itest

# load files
chunksize = 10 ** 4

train_chunks = pd.read_table(os.path.join(basepath, 'data/raw/train_v2.csv'),                              chunksize=chunksize,                              sep=',',                              index_col='id'
                            )

train = pd.concat(train_chunks)

# create a binary variable based on the target
train['is_default'] = (train.loss > 0).astype(np.int)

itrain, itest = get_stratified_sample(train, train.is_default, 0.4)

train_sample = train.iloc[itrain]
del train

print('Shape of the sample: ', (train_sample.shape))

features = train_sample.columns.drop(['is_default', 'loss'])

start_index = 760
end_index   = 770

train_sample.ix[:, start_index:end_index].hist(figsize=(16, 12), bins=50)
plt.savefig(os.path.join(basepath, 'reports/figures/feat_%s-%s'%(start_index, end_index)))

itrain, itest = get_stratified_sample(train_sample, train_sample.is_default, train_size=0.7, random_state=11)

X_train = train_sample.iloc[itrain][features]
X_test  = train_sample.iloc[itest][features]

y_train = train_sample.is_default.iloc[itrain]
y_test  = train_sample.is_default.iloc[itest]

class GoldenFeature(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X['f528-f527'] = X['f528'] - X['f527'] 
        X['f528-f274'] = X['f528'] - X['f274']
        return X

class TreeBasedSelection(object):
    def __init__(self, estimator, target, n_features_to_select=None):
        self.estimator            =  estimator
        self.n_features_to_select =  n_features_to_select
        self.target               =  target
        
    def fit(self, X, y=None):
        self.estimator.fit(X, self.target)
        
        self.importances = self.estimator.feature_importances_
        self.indices     = np.argsort(self.importances)[::-1]
        
        return self
    
    def transform(self, X):
        return X[:, self.indices[:self.n_features_to_select]]

class FeatureInteraction(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    @staticmethod
    def _combinations(features):
        return combinations(features, 2)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        features = map(str, list(range(X.shape[1])))
        interactions = []
        
        for comb in self._combinations(features):
            feat_1, feat_2 =  comb
            interactions.append(X[:, int(feat_2)] - X[:, int(feat_1)])
        
        return np.vstack(interactions).T

class VarSelect(BaseEstimator, TransformerMixin):
    def __init__(self, features, regexp_feature=r'.*-.*'):
        self.keys = [col for col in features if len(re.findall(regexp_feature, col)) > 0]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.keys]

cv = StratifiedKFold(y_train, n_folds=3, random_state=11)
score = 0
index = 0

for tr, ts in cv:
    print('Fold: %d'%(index))
    index += 1
    
    Xtr = X_train.iloc[tr]
    Xte = X_train.iloc[ts]
    
    ytr = y_train.iloc[tr]
    yte = y_train.iloc[ts]
    
    pipeline = Pipeline([
        ('feature_union', FeatureUnion([
                    ('golden_feature', GoldenFeature())
                ])),
        ('imputer', Imputer()),
        ('scaler', MinMaxScaler()),
        ('select', TreeBasedSelection(ExtraTreesClassifier(), ytr, n_features_to_select=30)),
#         ('select', TruncatedSVD(n_components=30)),
        ('union', FeatureUnion([
                    ('feature_interaction', FeatureInteraction())
                ])),
        ('model', RandomForestClassifier(n_estimators=25, n_jobs=2, random_state=5))
    ])

    pipeline.fit(Xtr, ytr)
    preds = pipeline.predict_proba(Xte)[:, 1]
    score += roc_auc_score(yte, preds)
    
print('CV scores ', score/len(cv))

preds = pipeline.predict_proba(X_test)[:, 1]
print('AUC score on unseen examples %f'%(roc_auc_score(y_test, preds)))





















class FeatureExtractor:
    def __init__(self, train, test):
        self.train = train
        self.test  = test
    
    def extract(self):
        self.round_values()
        self.create_features()
        
        return self.get_train(), self.get_test()
    
    def round_values(self):
        self.train = np.around(self.train, decimals=1)
        self.test  = np.around(self.test, decimals=1)
    
    def create_features(self):
        # feature based out of f1
        self.train['f1_cat'] = (self.train['f1'] < 140).astype(np.int)
        self.test['f1_cat']  = (self.test['f1'] < 140).astype(np.int)
        
        # feature based out of f9
        self.train['f9_cat'] = (self.train['f9'] < 140).astype(np.int)
        self.test['f9_cat']  = (self.test['f9'] < 140).astype(np.int)
        
        # feature based out of 10
        self.train['f10_cat'] = (self.train['f10'] < 140).astype(np.int)
        self.test['f10_cat']  = (self.test['f10'] < 140).astype(np.int)
        
        # feature out of f14
        self.train['f14_cat'] = (self.train['f14'] == 0.0).astype(np.int)
        self.test['f14_cat']  = (self.test['f14'] == 0.0).astype(np.int)
        
        # feature out of f6
        self.train['f6_cat'] = (self.train['f6'] < 2e4).astype(np.int)
        self.test['f6_cat']  = (self.test['f6'] < 2e4).astype(np.int)
         
    def get_train(self):
        return self.train
    
    def get_test(self):
        return self.test

feat = FeatureExtractor(train[train.columns[:12]], test[test.columns[:12]])
train_sub, test_sub = feat.extract()

train_sub.to_csv(os.path.join(basepath, 'data/processed/train_sub.csv'), index=False)
test_sub.to_csv(os.path.join(basepath, 'data/processed/test_sub.csv'), index=False)

train[['loss']].to_csv(os.path.join(basepath, 'data/processed/target.csv'), index=False)



