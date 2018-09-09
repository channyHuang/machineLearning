import csv
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import os
import sys
sys.path.append("/home/channy/Documents/Code/caffe/python")
sys.path.append("/home/channy/Documents/Code/caffe/python/caffe")
import caffe

#caffe train --solver=/home/channy/Documents/Code/machineLearning/solver.prototxt

import warnings 
with warnings.catch_warnings():
	warnings.filterwarnings("ignore",category=FutureWarning)
import h5py

def writeToHd5(trainData, testData):
	trainData = trainData.dropna()
	trainData['Image'] = trainData['Image'].apply(lambda im: np.fromstring(im, sep=' ') )
	X_train = np.vstack(trainData['Image'].values).astype(np.float32)
	y_train = trainData.drop('Image', axis=1).values.astype(np.float32)
	X_train, y_train = shuffle(X_train, y_train, random_state=42)
	
	testData = testData.dropna()
	testData['Image'] = testData['Image'].apply(lambda im: np.fromstring(im, sep=' ') )
	X_test = np.vstack(testData['Image'].values).astype(np.float32)
	X_test = X_test.reshape((X_test.shape[0],1,96,96))
	
	file = h5py.File("training.hd5", "w")
	file.create_dataset("data", data=X_train[:1600], compression="gzip", compression_opts=4)
	file.create_dataset("label", data=y_train[:1600], compression="gzip", compression_opts=4)
	file.close()
	with open("training.txt", "w") as f:
		f.write(os.getcwd() + "/" + "training.hd5")
		
	file = h5py.File("verify.hd5", "w")
	file.create_dataset("data", data=X_train[1600:], compression="gzip", compression_opts=4)
	file.create_dataset("label", data=y_train[1600:], compression="gzip", compression_opts=4)
	file.close()
	with open("verify.txt", "w") as f:
		f.write(os.getcwd() + "/" + "verify.hd5")
	
	file = h5py.File("test.hd5", "w")
	file.create_dataset("data", data=X_test, compression="gzip", compression_opts=4)
	file.close()
	with open("test.txt", "w") as f:
		f.write(os.getcwd() + "/" + "test.hd5")

def initData(trainData, testData):
	return [trainData, testData]

if __name__ == '__main__':
	trainData = pd.read_csv('./kaggle/Facial_Keypoints/training.csv')
	testData = pd.read_csv('./kaggle/Facial_Keypoints/test.csv')
	
	[trainData, testData] = initData(trainData, testData)
	#trainData.info()
	#testData.info()
	
	writeToHd5(trainData, testData)