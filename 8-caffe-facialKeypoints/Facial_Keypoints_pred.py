import sys
sys.path.append("/home/channy/Documents/Code/caffe/python")
sys.path.append("/home/channy/Documents/Code/caffe/python/caffe")
import caffe
import h5py
import numpy as np
import pandas as pd
import itertools

if __name__ == '__main__':
	
	file = h5py.File("test.h5", "r")
	X_test = file['data'][:]
	net = caffe.Net("./facial_keypoints_pred.prototxt", "./fkp_iter_15000.caffemodel", caffe.TEST)
	data = np.zeros([X_test.shape[0], 1, 1, 1])
	net.set_input_arrays(X_test.astype(np.float32), data.astype(np.float32))
	result = net.forward()
	y_pred = net.blobs['fc6'].data
	
	features = ["left_eye_center_x","left_eye_center_y","right_eye_center_x","right_eye_center_y",
    "left_eye_inner_corner_x","left_eye_inner_corner_y","left_eye_outer_corner_x",
    "left_eye_outer_corner_y","right_eye_inner_corner_x","right_eye_inner_corner_y",
    "right_eye_outer_corner_x","right_eye_outer_corner_y","left_eyebrow_inner_end_x",  
    "left_eyebrow_inner_end_y","left_eyebrow_outer_end_x","left_eyebrow_outer_end_y",
    "right_eyebrow_inner_end_x", "right_eyebrow_inner_end_y","right_eyebrow_outer_end_x", 
    "right_eyebrow_outer_end_y","nose_tip_x","nose_tip_y","mouth_left_corner_x","mouth_left_corner_y",
    "mouth_right_corner_x","mouth_right_corner_y","mouth_center_top_lip_x","mouth_center_top_lip_y",
	"mouth_center_bottom_lip_x","mouth_center_bottom_lip_y"]
	
	index = range(1, len(y_pred) * len(features) + 1)
	imageId = range(1, len(y_pred) + 1)
	imageName = range(1, len(y_pred)  * len(features) + 1)
	featureName = range(1, len(y_pred)  * len(features) + 1)
	idxList = itertools.product(imageId, features)
	i = 0
	for idx in idxList:
		imageName[i] = idx[0]
		featureName[i] = idx[1]
		i += 1
	y_pred = y_pred.reshape(len(y_pred) * 30, 1)
	y_pred = list(y_pred)
	res = pd.DataFrame({'ImageId':imageName, 'FeatureName':featureName, 'Location':y_pred})
	finalRes = pd.read_csv('IdLookupTable.csv', header = 0)
	finalRes = finalRes.drop('Location', axis = 1)
	finalRes = finalRes.merge(res, on=['ImageId', 'FeatureName'])
	finalRes = finalRes.drop(['ImageId', 'FeatureName'], axis = 1)
	finalRes.to_csv('result.csv', index = False)
	
	