# 數據讀取
import csv
import pandas as pd
import numpy as np
# 計時
import time
#from sklearn.linear_model.logistic import LogisticRegression
from sklearn import svm
# 数据分析显示统计图
from matplotlib import pyplot as plt 
# ML library
import keras
from keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization,Convolution2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.utils.np_utils import to_categorical

def initData(trainData, testData):
	# 查看数据大小、格式
	print(f"trainData size = {trainData.shape}")
	print(trainData.info)
	# 查看数据统计值
	print("trainData describe")
	disc_train = trainData.describe().T
	disc_test = testData.describe().T
	print(disc_train.iloc[1:10, :])
	# 查看统计图
	fig, ax_arr = plt.subplots(1, 2, figsize = (14, 4))
	fig.subplots_adjust(wspace = 0.25, hspace = 0.025)
	ax_arr = ax_arr.ravel()
	sets = iter([(disc_train, "training"), (disc_test, "testing")])
	for i, ax in enumerate(ax_arr):
		set_ = next(sets)
		ax.plot(set_[0].loc[:, "mean"], label = "Mean")
		ax.set_title("Mean of the {} features".format(set_[1]))
		ax.set_xlabel("Pixels")
		ax.set_ylabel("Mean")
		ax.set_xticks([0, 120, 250, 370, 490, 610, 720])
		ax.legend(loc="upper left", shadow = True, frameon = True, framealpha = 0.9)
		ax.set_ylim([0, 150])
	#plt.show()
	# 归一化
	trainNorm = trainData.iloc[:, 1:] / 255.0
	testNorm = testData / 255.0
	# 查看训练图集
	rand_indices = np.random.choice(trainNorm.shape[0], 64, replace = False)
	examples = trainNorm.iloc[rand_indices, :]
	fig, ax_arr = plt.subplots(8, 8, figsize = (6, 5))
	fig.subplots_adjust(wspace = 0.25, hspace = 0.025)
	ax_arr = ax_arr.ravel()
	for i, ax in enumerate(ax_arr):
		ax.imshow(examples.iloc[i, :].values.reshape(28, 28), cmap = "gray")
		ax.axis("off")
	#plt.show()
	
	return [trainData, testData]
	
if __name__ == '__main__':
    # 加载数据
    trainData = pd.read_csv('./kaggleData/Digit_recognizer/train.csv')
    testData = pd.read_csv('./kaggleData/Digit_recognizer/test.csv')
    [traindata, testdata] = initData(trainData, testData)
    target = traindata.iloc[:, 1:]
    label = traindata.iloc[:, 0]
    target = np.array(target)
    df_test = np.array(testData)
    # 歸一化 
    target=target.reshape(42000,28,28)
    df_test=df_test.reshape(28000,28,28)
    target = target/ target.max()
    df_test = df_test / df_test.max()
    label=np.array(label)
    label_cat=to_categorical(label,10)
    
    target=target.reshape(42000,28,28, 1)
    df_test=df_test.reshape(28000,28,28, 1)
    # 建立網絡
    classifier = Sequential()
    classifier.add(Convolution2D(filters = 128, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    # BN归一
    classifier.add(BatchNormalization())
    classifier.add(Convolution2D(filters = 128, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    # dropout 防止过拟合而抛弃部分数据
    classifier.add(Dropout(0.25))
    classifier.add(Convolution2D(filters =256, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Convolution2D(filters = 256, kernel_size = (3,3),padding = 'Same',
                 activation ='relu'))
    classifier.add(BatchNormalization())
    classifier.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
    classifier.add(Dropout(0.25))
    
    # flatten
    classifier.add(Flatten())
    # dense 全连接层
    classifier.add(Dense(256, activation = "relu"))
    classifier.add(Dropout(0.3))
    classifier.add(Dense(10, activation = "softmax"))
    classifier.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
    # 預測
    classifier.fit(target,label_cat,epochs=50)
    results=classifier.predict(df_test)
    print(results)
    results = np.argmax(results, axis = 1)
    print(results)
    results = pd.Series(results,name="Label")
    submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
    submission.to_csv("submission.csv",index=False,header=True)
