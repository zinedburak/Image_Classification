"""Burak Deniz S010031 Department of Computer Science"""


import cv2
import numpy as np
import os
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib
from scipy.cluster.vq import vq

clf, classes_names, stdSlr, k, voc = joblib.load("bovw.pkl")
clf_grid1, classes_names_grid1, stdSlr_grid1, k_grid1, voc_grid1 = joblib.load("bovw_grid1.pkl")
clf_grid2, classes_names_grid2, stdSlr_grid2, k_grid2, voc_grid2 = joblib.load("bovw_grid2.pkl")
clf_meanShift, classes_names_grid2_meanShift = joblib.load("bovw_meanshift.pkl")

test_path = 'dataset/test'  # Names are airplane, cars, faces, motorbikes

testing_names = os.listdir(test_path)

# Get path to all images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
image_classes = []
class_id = 0


# To make it easy to list all file names in a directory let us define a function
#
def imglist(path):
    return [os.path.join(path, f) for f in os.listdir(path)]


# Fill the placeholder empty lists with image path, classes, and add class ID number

for testing_name in testing_names:
    dir = os.path.join(test_path, testing_name)
    class_path = imglist(dir)
    image_paths += class_path
    image_classes += [class_id] * len(class_path)
    class_id += 1

# create description list for key-points,grid1,grid2
des_list = []
des_list_grid1 = []
des_list_grid2 = []


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


sift = cv2.xfeatures2d.SIFT_create()

for image_path in image_paths:
    im = cv2.imread(image_path)
    kpts, des = sift.detectAndCompute(im, None)
    des_list.append((image_path, des))

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

# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
descriptors_grid1 = des_list_grid1[0][1]
descriptors_grid2 = des_list_grid2[0][1]

for image_path, descriptor in des_list[0:]:
    descriptors = np.vstack((descriptors, descriptor))

for image_path, descriptor_grid1 in des_list_grid1[1:]:
    descriptors_grid1 = np.vstack((descriptors_grid1, descriptor_grid1))

for image_path, descriptor_grid2 in des_list_grid2[1:]:
    descriptors_grid2 = np.vstack((descriptors_grid2, descriptor_grid2))

# Calculate the histogram of features
# vq Assigns codes from a code book to observations.


test_features = np.zeros((len(image_paths), k), "float32")
test_features_grid1 = np.zeros((len(image_paths), k), "float32")
test_features_grid2 = np.zeros((len(image_paths), k), "float32")

for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1], voc)
    for w in words:
        test_features[i][w] += 1

for i in range(len(image_paths)):
    words_grid1, distance_grid1 = vq(des_list_grid1[i][1], voc_grid1)
    for w_grid1 in words_grid1:
        test_features_grid1[i][w_grid1] += 1

for i in range(len(image_paths)):
    words_grid2, distance_grid2 = vq(des_list_grid2[i][1], voc_grid2)
    for w_grid2 in words_grid2:
        test_features_grid2[i][w_grid2] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum((test_features > 0) * 1, axis=0)
idf = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurences + 1)), 'float32')

nbr_occurrences_grid1 = np.sum((test_features_grid1 > 0) * 1, axis=0)
idf_grid1 = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurrences_grid1 + 1)), 'float32')

nbr_occurrences_grid2 = np.sum((test_features_grid2 > 0) * 1, axis=0)
idf_grid2 = np.array(np.log((1.0 * len(image_paths) + 1) / (1.0 * nbr_occurrences_grid2 + 1)), 'float32')

test_features = stdSlr.transform(test_features)
test_features_grid1 = stdSlr_grid1.transform(test_features_grid1)
test_features_grid2 = stdSlr_grid2.transform(test_features_grid2)

# Report true class names so they can be compared with predicted classes
print("prediction and accuracy of key-points without hand made grid ")

true_class = [classes_names[i] for i in image_classes]
# Perform the predictions and report predicted class names.
predictions = [classes_names[i] for i in clf.predict(test_features)]

# Print the true class and Predictions
print("true_class =" + str(true_class))
print("prediction =" + str(predictions))

accuracy = accuracy_score(true_class, predictions)
print("accuracy = ", accuracy)
cm = confusion_matrix(true_class, predictions)
print(cm)




# print(len(clf_meanShift), "len of clf mean shift")
print("prediction and accuracy of grid-1 ")

# Perform the predictions and report predicted class names.
predictions = [classes_names[i] for i in clf_grid1.predict(test_features_grid1)]

# Print the true class and Predictions
# print("true_class =" + str(true_class))
# print("prediction =" + str(predictions))

accuracy = accuracy_score(true_class, predictions)
print("accuracy = ", accuracy)
cm = confusion_matrix(true_class, predictions)
print(cm)

print("prediction and accuracy of grid-2 ")
# Perform the predictions and report predicted class names.
predictions = [classes_names[i] for i in clf_grid2.predict(test_features_grid2)]

# Print the true class and Predictions
# print("true_class =" + str(true_class))
# print("prediction =" + str(predictions))

accuracy = accuracy_score(true_class, predictions)
print("accuracy = ", accuracy)
cm = confusion_matrix(true_class, predictions)
print(cm)


print("prediction and accuracy of MeanShift ")

predictions = [classes_names[i] for i in clf_meanShift.predict(test_features_grid1)]
accuracy = accuracy_score(true_class, predictions)
print("accuracy = ", accuracy)
cm = confusion_matrix(true_class, predictions)
print(cm)
