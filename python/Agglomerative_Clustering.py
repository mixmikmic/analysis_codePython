import sklearn.metrics as sm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as sm
from sklearn import datasets
from sklearn.cluster import AgglomerativeClustering

iris = datasets.load_iris()
x = pd.DataFrame(iris.data)
x.columns = ['SepalLength','SepalWidth','PetalLength','PetalWidth'] 

y = pd.DataFrame(iris.target)
y.columns = ['Targets']

iris = x[['SepalLength', 'PetalLength']]

model =AgglomerativeClustering(n_clusters=3)
model

model.fit(iris)
model.labels_

# Set the size of the plot
plt.figure(figsize=(24,10))

# Create a colormap
colormap = np.array(['red', 'lime', 'black'])

# Plot Original
plt.subplot(1, 2, 1)
plt.scatter(x.SepalLength, x.PetalLength, c="K", s=40)
plt.title('Original dataset')

# Plot the Model

plt.subplot(1, 2, 2)
plt.scatter(x.SepalLength, x.PetalLength, c=colormap[model.labels_], s=40)
plt.title('Agglomerative Clustering')


plt.show()

def confusion(y,labels):
    cm = sm.confusion_matrix(y, labels)
    return cm

confusion(y, model.labels_)

# Calculate purity 
def Purity(cm):
    M=[]
    S=0
    for i in cm:
        k = max(i)
        M.append(k)
    for i in M:
        S+=i
    Purity=S/150 
    return Purity

Purity(confusion(y, model.labels_))

