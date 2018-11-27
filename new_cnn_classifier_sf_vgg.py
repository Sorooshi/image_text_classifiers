import numpy as np 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.models import load_model


# Helper libraries
import os
import glob
import cv2
import json
from collections import OrderedDict
from sklearn.model_selection import train_test_split
#import pathlib


with open ('train_params.json', 'r') as fp:
    train_params = json.load(fp)

print("training_params:", train_params)

 
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
    imgs_data.append(image)

images_data = np.asarray(imgs_data)
print('images_data_shape:', images_data.shape)

# pre-processing the data (normalizing and centering data)
num_images, rows, cols , chs = images_data.shape

#images_data = images_data - 120.0
#images_data /= np.array(255.0)

train_images, test_images, train_labels, test_labels = train_test_split(images_data, labels, test_size=0.15, random_state=0)

'''
np.save("/home/soroosh/code/new_cnn_classifier/train_images_np", train_images)
np.save("/home/soroosh/code/new_cnn_classifier/train_labels_np", train_labels)
np.save("/home/soroosh/code/new_cnn_classifier/test_images_np", test_images)
np.save("/home/soroosh/code/new_cnn_classifier/test_labels_np", test_labels)
'''

'''
train_images = np.load("train_images_np.npy")
train_labels = np.load("train_labels_np.npy")
test_images = np.load("test_images_np.npy")
test_labels = np.load("test_labels_np.npy")

num_images, rows, cols , chs = train_images.shape
print("dimension:", train_images.shape)
'''

print("X_train:", train_images.shape)
print("X_test:", test_images.shape)

print("Y_train:", len(train_labels))
print("Y_test:", len(test_labels))


# Constructing the model:
   # model.add(Flatten(128, input_shape=(rows, cols, chs)))
   # model.add(Dense(128, input_shape=(none, rows, cols, chs)))
def create_model():
    model = Sequential()
    # 64 convlutional filters of size 3*3 (kernel_size) "input_shape just for first layer is necessary"
    model.add(Conv2D(64, (3,3), activation='relu', input_shape=(rows, cols, chs)))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf" )) #  data_format='channels_last'
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
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf" ))
    model.add(Dropout(0.25))

    print("3")

    model.add(Conv2D(512,(3,3), activation='relu'))
    model.add(Conv2D(512,(3,3), activation='relu'))
    model.add(Conv2D(512,(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), dim_ordering="tf" ))
    model.add(Dropout(0.25))

    print("4")

    
    model.add(Flatten())
    model.add(Dense(4094, activation='relu'))
    model.add(Dense(4094, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))

    print("5")

     
    # Optimizer
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    #adam = adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None)
    # other optimizer
    model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#     ['accuracy', 'sparse_top_k_categorical_accuracy']
    
    return model


''' 
train_acc_sf_vgg = OrderedDict([])
train_loss_sf_vgg = OrderedDict([])

test_acc_sf_vgg = OrderedDict([])
test_loss_sf_vgg = OrderedDict([])
'''

with open ("train_acc_sf_vgg.json", 'r') as fp:
   train_acc_sf_vgg = json.load(fp)

with open ("train_loss_sf_vgg.json", 'r') as fp:
   train_loss_sf_vgg = json.load(fp)

with open ("test_acc_sf_vgg.json", 'r') as fp:
   test_acc_sf_vgg = json.load(fp)

with open ("test_loss_sf_vgg.json", 'r') as fp:
   test_loss_sf_vgg = json.load(fp)


#for num_epochs in train_params['num_epochs']:
for num_epochs in [50, 75, 100] : #  [20, 30, 40]

    model = create_model()
    
    # Training the model
    model.fit(train_images, train_labels, epochs = num_epochs, batch_size=32,) # callbacks=[cp_callback]

    # train_eval:
    train_loss_sf_vgg[str(num_epochs)], train_acc_sf_vgg[str(num_epochs)] = model.evaluate(train_images, train_labels)

    # test_eval:
    test_loss_sf_vgg[str(num_epochs)], test_acc_sf_vgg[str(num_epochs)] = model.evaluate(test_images, test_labels)

   # prediction:
   # model.predict(test_images)

    print('Test accuracy:', test_acc_sf_vgg.items())
    print('Test loss:', test_loss_sf_vgg.items())

    model_name = "model_sf_vgg"+str(num_epochs)
    print("model name:", model_name)
    model.save(model_name+".h5")

    with open("test_acc_sf_vgg.json", 'w+') as fp:
        json.dump(test_acc_sf_vgg, fp)

    with open("test_loss_sf_vgg.json", 'w+') as fp:
        json.dump(test_loss_sf_vgg, fp)

    with open("train_acc_sf_vgg.json", "w+") as fp:
        json.dump(train_acc_sf_vgg, fp)

    with open("train_loss_sf_vgg.json", "w+") as fp:
        json.dump(train_loss_sf_vgg, fp)
