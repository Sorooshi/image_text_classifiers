import numpy as np 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import load_model
from keras.preprocessing.image import img_to_array 


# Helper libraries
import os
import glob
import cv2
import json
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


file_names = []
tmp_labels = []
 
# loading images:
dir_paths = ".\\multi_dataset\\[a-zA-Z]*"
directories = glob.glob(dir_paths)
print("directories:", directories, len(directories))

for directory in directories:
    image_paths = glob.glob(directory+"\\*.jpg")
    for i in range(len(image_paths)):
        file_names.append(image_paths[i])
        tmp_labels.append(image_paths[i].split("\\")[-2].split("_"))

print("len(file_names)",len(file_names)) # file_names,
print("len(tmp_labels)", len(tmp_labels)) # labels,

# Multi-label classes 
mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(tmp_labels)
class_names = mlb.classes_

print("class names:", class_names)
print("number of classes:", len(class_names))
                      
imgs_data = []
row_dim, col_dim = 224, 224

for file in file_names:
    image = cv2.imread(file)
    image = cv2.resize(image,(row_dim, col_dim))
    image = np.asarray(image)
    imgs_data.append(image)

images_data = np.asarray(imgs_data)
print('images_data_shape:', images_data.shape)

# pre-processing the data (normalizing and centering data)
num_images, rows, cols , chs = images_data.shape

#images_data = images_data - 120.0
#images_data /= np.array(255.0)

train_images, test_images, train_labels, test_labels = train_test_split(images_data, labels, test_size=0.15, random_state=0)

print("X_train:", train_images.shape)
print("X_test:", test_images.shape)

print("Y_train:", len(train_labels))
print("Y_test:", len(test_labels))


model = load_model("C:/Users/srsha/image_text_classifier/vgg_mc_ml/multi_classes_multi_labels/model_mL_vgg90.h5")


for i in range(len(test_images)):
    classes_prob = model.predict(np.expand_dims(test_images[i], axis=0))
    print("classes probability:", (classes_prob))
    index = np.argsort(classes_prob[::-1])
    print("index:", index)
    image_description = str(class_names[index[0][-1]]) + ": " + str(round(float(classes_prob[0][index[0][-1]]*100), 1)) + "% " + \
                        ", " + str(class_names[index[0][-2]]) + ": " + str(round(float(classes_prob[0][index[0][-2]])*100, 1))+"%"
    print("image_description:", image_description)
    #print(class_names[int(pred_class)])
    cv2.imshow(image_description, cv2.resize(test_images[i], (600,450)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
