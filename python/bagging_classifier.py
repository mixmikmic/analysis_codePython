import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('seaborn')

import numpy as np
import sklearn
import matplotlib
import pandas as pd
import sys
libraries = (('Matplotlib', matplotlib), ('Numpy', np), ('Pandas', pd))

print("Python Version:", sys.version, '\n')
for lib in libraries:
    print('{0} Version: {1}'.format(lib[0], lib[1].__version__))

import sys 
sys.path.append('../modules')
from decision_tree_classifier import decision_tree_classifier
import collections
import pandas as pd
import numpy as np

class bagging_classifier:
    
    def __init__(self, n_trees = 10, max_depth=None):
        """
        Bagging Classifier uses bootstrapping to generate n_trees different
        datasets and then applies a decision tree to each dataset. The final 
        prediction is an ensemble of all created trees.
        ---
        Params:
        n_trees (int): number of bootstrapped trees to grow for ensembling
        max_depth (int): maximum number of splits to make in each tree)
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
    
    def get_bagged_data(self, X, y):
        """
        Chooses random rows to populate a bootstrapped dataset, with replacement.
        Maintains the correlation between X and y
        ---
        Input: X, y (arrays)
        Outputs: randomized X,y (arrays)
        """
        index = np.random.choice(np.arange(len(X)),len(X))
        return X[index], y[index]
    
    def pandas_to_numpy(self, x):
        """
        Checks if the input is a Dataframe or series, converts to numpy matrix for
        calculation purposes.
        ---
        Input: X (array, dataframe, or series)
        Output: X (array)
        """
        if type(x) == type(pd.DataFrame()) or type(x) == type(pd.Series()):
            return x.as_matrix()
        if type(x) == type(np.array([1,2])):
            return x
        return np.array(x)
    
    def fit(self, X, y):
        """
        Generates the bootstrapped data then uses the decision tree
        class to build a model on each bootstrapped dataset. Each tree
        is stored as part of the model for later use.
        ---
        Input: X, y (arrays, dataframe, or series)
        """
        X = self.pandas_to_numpy(X)
        y = self.pandas_to_numpy(y)
        for _ in range(self.n_trees):
            bagX, bagy = self.get_bagged_data(X,y)
            new_tree = decision_tree_classifier(self.max_depth)
            new_tree.fit(bagX, bagy)
            self.trees.append(new_tree)
            
    def predict(self, X):
        """
        Uses the list of tree models built in the fit, doing a predict with each
        model. The final prediction uses the mode of all the trees predictions.
        ---
        Input: X (array, dataframe, or series)
        Output: Class ID (int)
        """
        X = self.pandas_to_numpy(X)
        self.predicts = []
        for tree in self.trees:
            self.predicts.append(tree.predict(X))
        self.pred_by_row = np.array(self.predicts).T
        
        ensemble_predict = []
        for row in self.pred_by_row:
            ensemble_predict.append(collections.Counter(row).most_common(1)[0][0])
        return ensemble_predict
    
    def score(self, X, y):
        """
        Uses the predict method to measure the accuracy of the model.
        ---
        In: X (list or array), feature matrix; y (list or array) labels
        Out: accuracy (float)
        """
        pred = self.predict(X)
        correct = 0
        for i,j in zip(y,pred):
            if i == j:
                correct+=1
        return float(correct)/float(len(y))

def get_data():
    from sklearn.datasets import load_iris
    iris = load_iris()
    return iris.data, iris.target

X,y = get_data()

from data_splitting import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

bc = bagging_classifier(n_trees=100)
bc.fit(X_train, y_train)

preds = bc.predict(X_test)
for i,j in zip(preds[10:40:2], bc.pred_by_row[10:40:2]):
    print(j,i)

bc.score(X_test,y_test)

accs = []
for n in range(1,100,5):
    bc = bagging_classifier(n_trees=n)
    bc.fit(X_train, y_train)
    accs.append(bc.score(X_test, y_test))

plt.plot(range(1,100,5),accs,'r')
plt.xlabel("Num. Trees")
plt.ylabel("Accuracy Score")
plt.title("Accuracy vs Num Trees (Mean Acc: %.3f)"%round(np.mean(accs),3));

from sklearn.datasets import load_wine
X = load_wine().data
y = load_wine().target

from data_splitting import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

bc = bagging_classifier(n_trees=100)
bc.fit(X_train, y_train)
bc.score(X_test, y_test)

from sklearn.dummy import DummyClassifier
dc = DummyClassifier()
dc.fit(X_train, y_train)
dc.score(X_test, y_test)

accs = []
for n in range(1,100,5):
    bc = bagging_classifier(n_trees=n)
    bc.fit(X_train, y_train)
    accs.append(bc.score(X_test, y_test))

plt.plot(range(1,100,5),accs,'r');
plt.xlabel("Num. Trees")
plt.ylabel("Accuracy Score");
plt.title("Accuracy vs Num Trees (Mean Acc: %.3f)"%round(np.mean(accs),3));



