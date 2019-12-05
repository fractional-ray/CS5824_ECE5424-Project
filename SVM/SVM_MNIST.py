import math, time
import numpy as np
import pandas as pd
from sklearn.svm import SVC

#load in datasets
MNIST_train_df = pd.read_csv('mnist_train.csv', sep=',')
MNIST_test_df = pd.read_csv('mnist_test.csv', sep=',')

F_MNIST_train_df = pd.read_csv('fashion-mnist_train.csv', sep=',')
F_MNIST_test_df = pd.read_csv('fashion-mnist_test.csv', sep=',')

#combine all data
tempM1 = MNIST_train_df.to_numpy(dtype=np.float64)
tempM2 = MNIST_test_df.to_numpy(dtype=np.float64)
allM = np.concatenate((tempM1, tempM2), axis=0)

tempF1 = F_MNIST_train_df.to_numpy(dtype=np.float64)
tempF2 = F_MNIST_test_df.to_numpy(dtype=np.float64)
allF = np.concatenate((tempF1, tempF2), axis=0)

#shuffle
np.random.shuffle(allM)
np.random.shuffle(allF)

#resplit into testing and training
MNIST_train = allM[:59999, :]
MNIST_test = allM[60000:, :]
F_MNIST_train = allF[:59999, :]
F_MNIST_test = allF[60000:, :]

X_tr = MNIST_train[:, 1:] # iloc ensures X_tr will be a dataframe
y_tr = MNIST_train[:, 0]
f_X_tr = F_MNIST_train[:, 1:] # iloc ensures X_tr will be a dataframe
f_y_tr = F_MNIST_train[:, 0]

X_test = MNIST_test[:, 1:]
y_test = MNIST_test[:, 0]
f_X_test = F_MNIST_test[:, 1:]
f_y_test = F_MNIST_test[:, 0]

#scale x values
X_tr = X_tr/255.0
X_test = X_test/255.0
f_X_tr = f_X_tr/255.0
f_X_test = f_X_test/255.0

##################################################
#C=1, sigmoid = test 1
print('Test 1: C=1, sigmoid')

clf_1M = SVC(kernel='sigmoid', C=1, gamma='auto')
clf_1F = SVC(kernel='sigmoid', C=1, gamma='auto')

#Test 1 - MNIST
start = time.time()
print('training MNIST test 1')
clf_1M.fit(X_tr, y_tr)

run_time = time.time() - start
print('Test 1 - MNIST- run in %.3f s' % run_time)
print("Test 1 MNIST Accuracy = %3.4f" % (clf_1M.score(X_test, y_test)))

#test 1 = FMNIST
start = time.time()
print('training FMNIST test 1')
clf_1F.fit(f_X_tr, f_y_tr)

run_time = time.time() - start
print('Test 1 - Fashion MNIST- run in %.3f s' % run_time)
print("Test 1 Fashion MNIST Accuracy = %3.4f" % (clf_1F.score(f_X_test, f_y_test)))
print(' ')

#####################
#C=10, poly = test 2
print('Test 2: C=10, poly')

clf_2M = SVC(kernel='poly', C=10, gamma='auto')
clf_2F = SVC(kernel='poly', C=10, gamma='auto')

#Test 1 - MNIST
start = time.time()
print('training MNIST test 1')
clf_2M.fit(X_tr, y_tr)

run_time = time.time() - start
print('Test 1 - MNIST- run in %.3f s' % run_time)
print("Test 1 MNIST Accuracy = %3.4f" % (clf_2M.score(X_test, y_test)))

#test 1 = FMNIST
start = time.time()
print('training FMNIST test 1')
clf_2F.fit(f_X_tr, f_y_tr)

run_time = time.time() - start
print('Test 1 - Fashion MNIST- run in %.3f s' % run_time)
print("Test 1 Fashion MNIST Accuracy = %3.4f" % (clf_2F.score(f_X_test, f_y_test)))
print(' ')

#####################
#C=100, poly = test 3
print('Test 3: C=100, poly')

clf_3M = SVC(kernel='poly', C=100, gamma='auto')
clf_3F = SVC(kernel='poly', C=100, gamma='auto')

#Test 1 - MNIST
start = time.time()
print('training MNIST test 1')
clf_3M.fit(X_tr, y_tr)

run_time = time.time() - start
print('Test 1 - MNIST- run in %.3f s' % run_time)
print("Test 1 MNIST Accuracy = %3.4f" % (clf_3M.score(X_test, y_test)))

#test 1 = FMNIST
start = time.time()
print('training FMNIST test 1')
clf_3F.fit(f_X_tr, f_y_tr)

run_time = time.time() - start
print('Test 1 - Fashion MNIST- run in %.3f s' % run_time)
print("Test 1 Fashion MNIST Accuracy = %3.4f" % (clf_3F.score(f_X_test, f_y_test)))
print(' ')

#####################
#C=10, rbf = test 4
print('Test 4: C=10, rbf')

clf_4M = SVC(kernel='rbf', C=10, gamma='auto')
clf_4F = SVC(kernel='rbf', C=10, gamma='auto')

#Test 1 - MNIST
start = time.time()
print('training MNIST test 1')
clf_4M.fit(X_tr, y_tr)

run_time = time.time() - start
print('Test 1 - MNIST- run in %.3f s' % run_time)
print("Test 1 MNIST Accuracy = %3.4f" % (clf_4M.score(X_test, y_test)))

#test 1 = FMNIST
start = time.time()
print('training FMNIST test 1')
clf_4F.fit(f_X_tr, f_y_tr)

run_time = time.time() - start
print('Test 1 - Fashion MNIST- run in %.3f s' % run_time)
print("Test 1 Fashion MNIST Accuracy = %3.4f" % (clf_4F.score(f_X_test, f_y_test)))
print(' ')

#####################
#C=100, rbf = test 5
print('Test 5: C=100, rbf')

clf_5M = SVC(kernel='rbf', C=100, gamma='auto')
clf_5F = SVC(kernel='rbf', C=100, gamma='auto')

#Test 1 - MNIST
start = time.time()
print('training MNIST test 1')
clf_5M.fit(X_tr, y_tr)

run_time = time.time() - start
print('Test 1 - MNIST- run in %.3f s' % run_time)
print("Test 1 MNIST Accuracy = %3.4f" % (clf_5M.score(X_test, y_test)))

#test 1 = FMNIST
start = time.time()
print('training FMNIST test 1')
clf_5F.fit(f_X_tr, f_y_tr)

run_time = time.time() - start
print('Test 1 - Fashion MNIST- run in %.3f s' % run_time)
print("Test 1 Fashion MNIST Accuracy = %3.4f" % (clf_5F.score(f_X_test, f_y_test)))
print(' ')

