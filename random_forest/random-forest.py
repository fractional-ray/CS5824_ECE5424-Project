import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 
import sys

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

#
# Uncomment for MNIST
#
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

clf = RandomForestClassifier(n_estimators=100,max_depth=10)
x_train = np.array(x_train)
y_train = np.array(y_train)


nsamples, nx, ny = x_train.shape
x_train = x_train.reshape((nsamples,nx*ny))

nsamples, nx, ny = x_test.shape
test_x = x_test.reshape((nsamples,nx*ny))


clf.fit(x_train, y_train)

name = sys.argv[1]
predicted = clf.predict(test_x)
print( accuracy_score(y_test, predicted))
cf = confusion_matrix(y_test, predicted)


sns.heatmap(cf,annot=False,cbar=False,cmap="Blues")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix Gini Fashion MNIST')
plt.savefig(name + ".png")
