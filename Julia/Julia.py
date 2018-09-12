import csv
import pandas as pd
import numpy as np
from sklearn import neighbors
import os
import cv2
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.utils import shuffle
import subprocess

import sys
sys.path.append("/home/channy/Documents/Code/caffe/python")
sys.path.append("/home/channy/Documents/Code/caffe/python/caffe")
import caffe

#caffe train --solver=/home/channy/Documents/Code/machineLearning/solver.prototxt

import warnings 
with warnings.catch_warnings():
	warnings.filterwarnings("ignore",category=FutureWarning)
import h5py

def runShell(file):
	p = subprocess.Popen(file, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
	while p.poll() == None:
		line = p.stdout.readline()
		print line 

def transferImage(imgDir):
	fileList = os.listdir(imgDir)
	X = np.zeros((len(fileList), 400))
	i = 0
	for filename in fileList:
		img = cv2.imread(imgDir + "/" + filename, cv2.IMREAD_GRAYSCALE)
		imgData = img.reshape(1, 400) #20*20
		X[i, :] = imgData
		i += 1
	return X

def char2int(labelList):
	y = np.zeros(len(labelList))
	i = 0
	for c in labelList:
		asciiVal = ord(c)
		if (asciiVal <= 57):
			asciiVal -= 47
		else:
			asciiVal -= 54
		y[i] = asciiVal
		i += 1
	return y

def int2char(labelList):
	y = range(len(labelList))
	i = 0
	for c in labelList:
		if (c <= 9):
			c += 47
		else:
			c += 54
		y[i] = chr((int)(c))
		i += 1
	return y

def initData():
	t = time.time()
	X_train = transferImage('./trainResized')
	X_test = transferImage('./testResized')
	y_train = pd.read_csv('./trainLabels.csv')
	y_train = char2int(y_train['Class'])
	trainData = [y_train, X_train]
	testData = X_test
	print('read data time : %f'%(time.time() - t))
	return [X_train, y_train, testData]

#seems that this model doesn't work, but if k change from 5 to 62, rate will increase from 0.02 to 0.05 @_@
#only k and distance can effect this algorithm, right?
def knn(X_train, y_train, testData):
	t = time.time()
	k = 62
	weights = ['uniform', 'distance']
	clf = neighbors.KNeighborsClassifier(k, 'distance', matric='minkowski')
	clf.fit(X_train, y_train)
	result = clf.predict(testData)
	result = int2char(result)
	index = range(6284,  6284 + len(result))
	finalRes = pd.DataFrame({'ID':index, 'Class':result})
	finalRes = finalRes[['ID', 'Class']]
	finalRes.to_csv('result.csv', index = False)
	print('solver time : %f'%(time.time() - t))

#n_estimators works only, other params, emmm......
#also works bad
'''
read data time : 0.512160
time : 1095.250432
(cpu only -.-)
'''
def RandomForest(X_train, y_train, testData):
	t = time.time()
	rfr = RandomForestClassifier(n_estimators = 1000)
	X = X_train
	Y = y_train
	rfr.fit(X, Y)
	result = rfr.predict(testData)
	result = int2char(result)
	index = range(1, len(result) + 1)
	finalRes = pd.DataFrame({'ID':index, 'Class':result})
	finalRes = finalRes[['ID', 'Class']]
	finalRes.to_csv('result.csv', index = False)
	print('time : %f'%(time.time() - t))	

#kernel, C
#next time, try to write kernel by self
#even worse than the above two...... Is there anything wrong with my code?
def svmClassify(X_train, y_train, testData):
	t = time.time()
	X = X_train
	Y = y_train
	svc = svm.SVC(kernel = 'rbf', C = 1)
	svc.fit(X, Y)
	result = svc.predict(testData)
	result = int2char(result)
	index = range(1, len(result) + 1)
	finalRes = pd.DataFrame({'ID':index, 'Class':result})
	finalRes = finalRes[['ID', 'Class']]
	finalRes.to_csv('result.csv', index = False)
	print('time : %f'%(time.time() - t))

#have no choice but try CNN
#reference: http://ankivil.com/kaggle-first-steps-with-julia-chars74k-first-place-using-convolutional-neural-networks/
#thanks for all peoples who share their experience
def writeToHd5(X_train, y_train, testData):
	t = time.time()
	X_train, y_train = shuffle(X_train, y_train, random_state=42)
	X_train = X_train.reshape((X_train.shape[0], 1, 20, 20))
	testData = testData.reshape((testData.shape[0], 1, 20, 20))
	
	file = h5py.File("training.hd5", "w")
	file.create_dataset("data", data=X_train[:3000], compression="gzip", compression_opts=4)
	file.create_dataset("label", data=y_train[:3000], compression="gzip", compression_opts=4)
	file.close()
	with open("training.txt", "w") as f:
		f.write(os.getcwd() + "/" + "training.hd5")

	file = h5py.File("verify.hd5", "w")
	file.create_dataset("data", data=X_train[3000:], compression="gzip", compression_opts=4)
	file.create_dataset("label", data=y_train[3000:], compression="gzip", compression_opts=4)
	file.close()
	with open("verify.txt", "w") as f:
		f.write(os.getcwd() + "/" + "verify.hd5")
	
	file = h5py.File("test.hd5", "w")
	file.create_dataset("data", data=testData, compression="gzip", compression_opts=4)
	file.close()
	with open("test.txt", "w") as f:
		f.write(os.getcwd() + "/" + "test.hd5")
	print('write to hd5 time : %f'%(time.time() - t))

def CNN(X_train, y_train, testData):
	t = time.time()
	writeToHd5(X_train, y_train, testData)
	runShell("./cmd_julia.sh")
	'''
	file = h5py.File("test.hd5", "r")
	X_test = file['data'][:]
	net = caffe.Net("julia_pred.prototxt", "tmp_iter_2000.caffemodel", caffe.TEST)
	data = np.zeros([X_test.shape[0], 1, 1, 1])
	net.set_input_arrays(X_test.astype(np.float32), data.astype(np.float32))
	result = net.forward()
	y_pred = net.blobs['fc6'].data
	finalRes = pd.DataFrame({'ID':index, 'Class':y_pred})
	finalRes = finalRes[['ID', 'Class']]
	finalRes.to_csv('result.csv', index = False)
	print('cnn time : %f'%(time.time() - t))
	'''

if __name__ == '__main__':
	[X_train, y_train, testData] = initData()
	#print X_train.shape #6283
	
	#knn(trainData, testData)
	#RandomForest(X_train, y_train, testData)
	#svmClassify(X_train, y_train, testData)
	CNN(X_train, y_train, testData)
	#fix it later
	'''
	7fcb38136000-7fcb3819f000 rw-p 00000000 00:00 0 *** Aborted at 1536570773 (unix time) try "date -d @1536570773" if you are using GNU date ***
PC: @     0x7fcb46876428 gsignal
*** SIGABRT (@0x3e80000378c) received by PID 14220 (TID 0x7fcb48263ac0) from PID 14220; stack trace: ***
    @     0x7fcb468764b0 (unknown)
    @     0x7fcb46876428 gsignal
    @     0x7fcb4687802a abort
    @     0x7fcb468b87ea (unknown)
    @     0x7fcb468c137a (unknown)
    @     0x7fcb468c553c cfree
    @     0x7fcb47c770d2 boost::detail::sp_counted_impl_p<>::dispose()
    @     0x7fcb47cc922a boost::detail::sp_counted_impl_p<>::dispose()
    @           0x40e632 caffe::Net<>::~Net()
    @     0x7fcb47cbf242 boost::detail::sp_counted_impl_p<>::dispose()
    @     0x7fcb47ded845 caffe::SGDSolver<>::~SGDSolver()
    @           0x40aca1 train()
    @           0x406fa0 main
    @     0x7fcb46861830 __libc_start_main
    @           0x4077c9 _start
    @                0x0 (unknown)
	'''