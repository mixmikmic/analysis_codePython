get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt

from IPython.html.widgets import interact

from sklearn import datasets

digits = datasets.load_digits()

def browse_images(digits):
    n = len(digits.images)
    def view_image(i):
        plt.imshow(digits.images[i], cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Training: %s' % digits.target[i])
        plt.show()
    interact(view_image, i=(0,n-1))

browse_images(digits)

