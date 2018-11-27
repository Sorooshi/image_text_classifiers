import numpy as np
import keras
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
# from keras.models import load_weights

# Helper libraries
import os
import glob
import cv2
import json
from sklearn.model_selection import train_test_split

 
# loading images:
doctors_path = "./new_dataset/medical/doctor/*.jpg"
doctor_files = glob.glob(doctors_path)
doctor_labels = [0 for _ in range(len(doctor_files))]
print("doctor:", len(doctor_labels))

nurses_path = "./new_dataset/medical/nurse/*.jpg"
nurse_files = glob.glob(nurses_path)
nurse_labels = [1 for _ in range(len(nurse_files))]
print("nurse",len(nurse_labels))

dentists_path = "./new_dataset/medical/dentist/*.jpg"
dentist_files = glob.glob(dentists_path)
dentist_labels = [2 for _ in range(len(dentist_files))]
print("dentist:", len(dentist_labels))
drills_path = "./new_dataset/tools/drill/*.jpg"
drill_files = glob.glob(drills_path)
drill_labels = [3 for _ in range(len(drill_files))]
print("drill:", len(drill_labels))

hammer_path = "./new_dataset/tools/hammer/*.jpg"
hammer_files = glob.glob(hammer_path)
hammer_labels = [4 for _ in range(len(hammer_files))]
print("hammer:", len(hammer_labels))

pliers_path = "./new_dataset/tools/pliers/*.jpg"
pliers_files = glob.glob(pliers_path)
pliers_labels = [5 for _ in range(len(pliers_files))]
print("pliers:", len(pliers_labels))

screwdriver_path = "./new_dataset/tools/screwdriver/*.jpg"
screwdriver_files = glob.glob(screwdriver_path)
screwdriver_labels = [6 for _ in range(len(screwdriver_files))]
print("screwdrivers:", len(screwdriver_labels))

file_names = doctor_files + nurse_files + dentist_files + drill_files + hammer_files + pliers_files + screwdriver_files
labels = doctor_labels + nurse_labels + dentist_labels + drill_labels + hammer_labels + pliers_labels + screwdriver_labels

print("len(file_names)",len(file_names)) # file_names,
print("len(labels)", len(labels)) # labels,


class_names = {0:"doctor", 1:"nurse", 2:"dentist", 3: "drill", 4:"hammer", 5:"pliers", 6:"screwdriver"}
print("class names:", class_names)
print("number of classes:", len(class_names))

imgs_data = []
for file in file_names:
    image = cv2.imread(file)
    image = cv2.resize(image, (224,224))
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


# wieghts = load_weights("weights_sf_vgg100.h5")
model = load_model("model_sf_vgg100.h5")

# results = model.predict(test_images)
# print("results:", results)

for i in range(len(test_images)):
    pred_class = model.predict_classes(np.expand_dims(test_images[i], axis=0))
    pred_prob = np.asarray(model.predict_proba(np.expand_dims(test_images[i], axis=0)))
    print("pred_prob:", pred_prob)
    image_description = str(class_names[int(pred_class)]) + ": " + str(float(np.max(pred_prob, axis=1)*100)) + "%"
    print("image_description:", image_description)
    #print(class_names[int(pred_class)])
    cv2.imshow(image_description, cv2.resize(test_images[i], (600,400)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

