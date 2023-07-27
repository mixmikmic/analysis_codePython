import numpy as np

import matplotlib.pyplot as plt

get_ipython().magic('matplotlib inline')

def logistic_regression(data, target):
    beta = beta_old = np.random.rand(data.shape[1] + 1)
    d = np.ndarray((data.shape[0], data.shape[1] + 1))
    d[:, 0] = 1
    d[:, 1:] = data
    d = d.T
    t = target.T
    while True:
        beta_old = np.array(beta)
        beta -= newton_step(d, t, beta)
        print(np.linalg.norm(beta - beta_old))
        if np.linalg.norm(beta - beta_old) < 0.01:
            break
    return beta
    
def newton_step(data, target, beta):
    p = sigmoid(beta.T @ data)
    jac = data @ (target - p)
    weights = np.eye(data.shape[1]) * p * (1-p)
    hessian = -data @ weights @ data.T
    return np.linalg.inv(hessian) @ jac
    
def sigmoid(x):
    return np.exp(x)/(1 + np.exp(x))

import sklearn.datasets

data, target = sklearn.datasets.make_classification(n_samples=100, n_features=2, n_redundant=0)

beta = logistic_regression(data, target)

plt.plot(data[np.where(target==1), 0], data[np.where(target==1), 1], 'bx')
plt.plot(data[np.where(target==0), 0], data[np.where(target==0), 1], 'rx')
CS = plt.contour(X, Y, f(X,Y))
plt.clabel(CS, inline=1, fontsize=10)

data.shape

x = np.linspace(-4, 4, 1000)
y = np.linspace(-4, 4, 1000)

X, Y = np.meshgrid(x, y)

beta@[1, -.8, 0]

def f(x1, x2):
    X1 = np.asarray(x1.flat)
    X2 = np.asarray(x2.flat)
    d = np.ones((X1.shape[0], 3))
    d[:, 1] = X1
    d[:, 2] = X2
    return ((beta @ d.T)).reshape(x1.shape[0], x1.shape[1])

plt.contour(X, Y, f(X,Y))



