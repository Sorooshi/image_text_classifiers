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
dir_paths = "/home/soroosh/code/new_cnn_classifier/multi_dataset/[a-zA-Z]*"
directories = glob.glob(dir_paths)
print("directories:", directories, len(directories))

for directory in directories:
    image_paths = glob.glob(directory +"/*.jpg")
    for i in range(len(image_paths)):
        file_names.append(image_paths[i])
        tmp_labels.append(image_paths[i].split("/")[-2].split("_"))
                         
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


# Constructing the model:
def create_model():
    model = Sequential()
    # 64 convlutional filters of size 3*3 (kernel_size) "input_shape just for first layer is necessary"
    model.add(Conv2D(64, (3,3), activation='relu', input_shape=(rows, cols, chs)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf")) #  data_format='channels_last'
    model.add(Dropout(0.25))

    print("0")
    
    model.add(Conv2D(128,(3,3), activation='relu'))
    model.add(Conv2D(128, (3,3), activation='relu'))
    model.add(MaxPooling2D(data_format="channels_last", pool_size=(2, 2)))
    model.add(Dropout(0.25))

    print("1")

    model.add(Conv2D(256,(3,3), activation='relu'))
    model.add(Conv2D(256,(3,3), activation='relu'))
    model.add(Conv2D(256,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))
    model.add(Dropout(0.25))

    print("2")

    model.add(Conv2D(512,(3,3), activation='relu'))
    model.add(Conv2D(512,(3,3), activation='relu'))
    model.add(Conv2D(512,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))
    model.add(Dropout(0.25))

    print("3")

    model.add(Conv2D(512,(3,3), activation='relu'))
    model.add(Conv2D(512,(3,3), activation='relu'))
    model.add(Conv2D(512,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf"))
    model.add(Dropout(0.25))

    print("4")
    
    model.add(Flatten())
    model.add(Dense(4094, activation='relu'))
    model.add(Dense(4094, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(6, activation='sigmoid'))

    print("5")
     
    # Optimizer
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    #adam = adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None)
    # other optimizer
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    
    return model


train_mL_vgg = OrderedDict([])
test_mL_vgg = OrderedDict([])


#for num_epochs in train_params['num_epochs']:
for num_epochs in [1, 15, 30, 60, 90] : #  [20, 30, 40]  [50, 75, 100]
    model = create_model()    

    # Training the model
    model.fit(train_images, train_labels, epochs = num_epochs, batch_size=32,) # callbacks=[cp_callback]

    # train_eval:
    train_mL_vgg[str(num_epochs)] = model.evaluate(train_images, train_labels)

    # test_eval:
    test_mL_vgg[str(num_epochs)] = model.evaluate(test_images, test_labels)

    print('Train accuracy:', train_mL_vgg.items())
    print('Test loss:', test_mL_vgg.items())

    model_name = "model_mL_vgg"+str(num_epochs)
    print("model name:", model_name)
    if num_epochs >= 50:
        model.save(model_name+".h5")

    with open("test_mL_vgg.json", 'w+') as fp:
        json.dump(test_mL_vgg, fp)

    with open("train_mL_vgg.json", "w+") as fp:
        json.dump(train_mL_vgg, fp)




