# setup environment
get_ipython().magic('matplotlib inline')
import mxnet as mx
import numpy as np
import os
import urllib
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from images2gif import writeGif
import logging
logging.basicConfig(level=logging.DEBUG)

if not os.path.exists('deep3d-0050.params'):
    urllib.urlretrieve('http://homes.cs.washington.edu/~jxie/download/deep3d-0050.params', 'deep3d-0050.params')
model = mx.model.FeedForward.load('deep3d', 50, mx.gpu(0))

# import urllib
# urllib.urlretrieve('url/to/image.jpg', 'demo.jpg')

shape = (384, 160)
img = cv2.imread('demo.jpg')
raw_shape = (img.shape[1], img.shape[0])
img = cv2.resize(img, shape)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()

X = img.astype(np.float32).transpose((2,0,1))
X = X.reshape((1,)+X.shape)
test_iter = mx.io.NDArrayIter({'left': X, 'left0':X})
Y = model.predict(test_iter)

right = np.clip(Y.squeeze().transpose((1,2,0)), 0, 255).astype(np.uint8)
right = Image.fromarray(cv2.cvtColor(right, cv2.COLOR_BGR2RGB))
left = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
writeGif('demo.gif', [left, right], duration=0.08)



