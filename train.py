"""Burak Deniz S010031 Department of Computer Science"""

import glob

import numpy as np
import cv2
import os
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from joblib import dump
from sklearn.cluster import MeanShift

# We take the labels from the folder names in train folder so everything must be in order

train_path = 'dataset/train'

training_names = os.listdir(train_path)

image_paths = []
image_classes = []
class_id = 0


def imgList(path):
    return [os.path.join(path, f) for f in os.listdir(path)]


for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = imgList(dir)
    image_paths += class_path
    image_classes += [class_id] * len(class_path)
    class_id += 1

des_list = []
des_list_grid1 = []
des_list_grid2 = []

sift = cv2.xfeatures2d.SIFT_create()


def create_grid(image, size):
    cp_image = np.copy(image)
    height, width, _ = cp_image.shape
    key_points = []
    # print(width, " = w", height, " = h")
    for pixelh in range(size, height + 1, size):
        for pixelw in range(size, width + 1, size):
            key_points.append(cv2.KeyPoint(pixelw, pixelh, _size=size))
    """
    cp_image = cv2.drawKeypoints(cp_image, key_points, cp_image)
    cv2.imshow('grid', cp_image)
    cv2.waitKey(0)
    """
    return key_points


for image_path in image_paths:
    image = cv2.imread(image_path)
    """Using the key points from sift detect """
    kpts, des = sift.detectAndCompute(image, None)
    des_list.append((image_path, des))
    """For visual understanding"""
    """
    image = cv2.drawKeypoints(image,kpts,image)
    cv2.imshow('keypoints',image)
    cv2.waitKey(0)
    """
for image_path in image_paths:
    image = cv2.imread(image_path)
    """Creating grid 1 that takes every 20 pixels as key points and computes their sift """
    key_points_grid1 = create_grid(image, 20)
    des_grid1 = sift.compute(image, key_points_grid1)
    des_list_grid1.append((image_path, des_grid1[1]))


for image_path in image_paths:
    image = cv2.imread(image_path)
    """Creating grid 2 that takes every 10 pixels as key point and computes their sift"""
    key_points_grid2 = create_grid(image, 10)
    des_grid2 = sift.compute(image, key_points_grid2)
    des_list_grid2.append((image_path, des_grid2[1]))

descriptors = des_list[0][1]
descriptors_grid1 = des_list_grid1[0][1]
descriptors_grid2 = des_list_grid2[0][1]

for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))

for image_path, descriptor_grid1 in des_list_grid1[1:]:
    descriptors_grid1 = np.vstack((descriptors_grid1, descriptor_grid1))

for image_path, descriptor_grid2 in des_list_grid2[1:]:
    descriptors_grid2 = np.vstack((descriptors_grid2, descriptor_grid2))

print("Check point 1")

descriptors_float = descriptors.astype(float)
descriptors_grid1_float = descriptors_grid1.astype(float)
descriptors_grid2_float = descriptors_grid2.astype(float)

# change the key for more experiment
k = 50
voc, variance = kmeans(descriptors_float, k, 1)
voc_grid1, variance_grid1 = kmeans(descriptors_grid1_float, k, 1)
voc_grid2, variance_grid2 = kmeans(descriptors_grid2_float, k, 1)

im_features = np.zeros((len(image_paths), k), "float32")
im_features_grid1 = np.zeros((len(image_paths), k), "float32")
im_features_grid2 = np.zeros((len(image_paths), k), "float32")

for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1], voc)
    for w in words:
        im_features[i][w] += 1

for i in range(len(image_paths)):
    words_grid1, distance_grid1 = vq(des_list_grid1[i][1], voc_grid1)
    for w_grid1 in words_grid1:
        im_features_grid1[i][w_grid1] += 1

for i in range(len(image_paths)):
    words_grid2, distance_grid2 = vq(des_list_grid2[i][1], voc_grid2)
    for w_grid2 in words_grid2:
        im_features_grid2[i][w_grid2] += 1

print("Checkpoint 2 ")

nbr_occurrences = np.sum((im_features > 0) * 1, axis=0)
idf = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurrences + 1)), 'float32')

nbr_occurrences_grid1 = np.sum((im_features_grid1 > 0) * 1, axis=0)
idf_grid1 = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurrences_grid1 + 1)), 'float32')

nbr_occurrences_grid2 = np.sum((im_features_grid2 > 0) * 1, axis=0)
idf_grid2 = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurrences_grid2 + 1)), 'float32')

print("Check Point 3")
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

stdSlr_grid1 = StandardScaler().fit(im_features_grid1)
im_features_grid1 = stdSlr_grid1.transform(im_features_grid1)

stdSlr_grid2 = StandardScaler().fit(im_features_grid2)
im_features_grid2 = stdSlr_grid2.transform(im_features_grid2)

print("Check Point 4 ")

clf = LinearSVC(max_iter=10000)
clf.fit(im_features, np.array(image_classes))


clf_grid1 = LinearSVC(max_iter=10000)
clf_grid1.fit(im_features_grid1, np.array(image_classes))

clf_grid2 = LinearSVC(max_iter=10000)
clf_grid2.fit(im_features_grid2, np.array(image_classes))

print("Check point mean shift ")

clf_meanShift = MeanShift(bandwidth=50,max_iter=10000)
"""Change im_features_grid2 for more experiment"""
clf_meanShift.fit(im_features_grid2,np.array(image_classes))



dump((clf, training_names, stdSlr, k, voc), "bovw.pkl", compress=3)
dump((clf_grid1, training_names, stdSlr_grid1, k, voc_grid1), "bovw_grid1.pkl", compress=3)
dump((clf_grid2, training_names, stdSlr_grid2, k, voc_grid2), "bovw_grid2.pkl", compress=3)

dump((clf_meanShift, training_names),"bovw_meanshift.pkl",compress=3)

print("burak deniz ")
