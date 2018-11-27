import tensorflow as tf
from tensorflow import keras
import glob

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2

''' 
daisies_path = "./datasets/flower_photos/daisy/*.jpg"
daisy_files = glob.glob(daisies_path)
daisy_label = [0 for i in range(len(daisy_files))]
print(len(daisy_label))
# labels.append(daisy_label)

dandelions_path = "./datasets/flower_photos/dandelion/*.jpg"
dandelion_files = glob.glob(dandelions_path)
dandelion_labels = [1 for i in range(len(dandelion_files))]
print(len(dandelion_labels))
# labels.append(dandelion_labels)

roses_path = "./datasets/flower_photos/roses/*.jpg"
ros_files = glob.glob(roses_path)
ros_labels = [2 for i in range(len(ros_files))]
print(len(ros_labels))
'''


doctors_path = "./datasets/medical/doctor/*.jpg"
doctor_files = glob.glob(doctors_path)
doctor_labels = [0 for _ in range(len(doctor_files))]
print("doctor:", len(doctor_labels))

nurses_path = "./datasets/medical/nurse/*.jpg"
nurse_files = glob.glob(nurses_path)
nurse_labels = [1 for _ in range(len(nurse_files))]
print("nurse",len(nurse_labels))

dentists_path = "./datasets/medical/dentist/*.jpg"
dentist_files = glob.glob(dentists_path)
dentist_labels = [2 for _ in range(len(dentist_files))]
print("dentist:", len(dentist_labels))

drills_path = "./datasets/tools/drill/*.jpg"
drill_files = glob.glob(drills_path)
drill_labels = [3 for _ in range(len(drill_files))]
print("drill:", len(drill_labels))

hammer_path = "./datasets/tools/hammer/*.jpg"
hammer_files = glob.glob(hammer_path)
hammer_labels = [4 for _ in range(len(hammer_files))]
print("hammer:", len(hammer_labels))

pliers_path = "./datasets/tools/pliers/*.jpg"
pliers_files = glob.glob(pliers_path)
pliers_labels = [5 for _ in range(len(pliers_files))]
print("pliers:", len(pliers_labels))

screwdriver_path = "./datasets/tools/screwdriver/*.jpg"
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

images_data = []
for file in file_names:
    image = cv2.imread(file)
    images_data.append(image)

print('image_data_shape:', np.array(images_data).shape)

images_data = np.asarray(images_data)
# images_data /= 255.0

''' 
plt.figure(figsize=(1,3))
for i in range(3):
    plt.subplot(1,3,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images_data[i]) # images_data[i], cmap=plt.cm.gray
    plt.show()
    plt.xlabel(class_names[int(labels[i])])
 
'''

from sklearn.model_selection import train_test_split
train_images, test_images, train_labels, test_labels = train_test_split(images_data, labels, test_size=0.4, random_state=0)

print("train_images:", train_images.shape)
# train_images_mean = np.mean(train_images)
train_images = train_images - np.mean(train_images)
# train_images /= np.array(255.0)
test_images = test_images - np.mean(test_images)


# train_images = train_images.astype(np.float32)
# test_images = test_images.astype(np.float32)

print("X_train:", train_images.shape)
print("X_test:", test_images.shape)

print("Y_train:", len(train_labels))
print("Y_test:", len(test_labels))


# Constructing tge model:
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(256, 256, 3)),
    keras.layers.Dense(512, activation=tf.nn.relu),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(7, activation=tf.nn.softmax)
])

# Optimizer
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training the model
model.fit(train_images, train_labels, epochs=3) # train_images, train_labels


# Predictions:
predictions = model.predict(test_images)
# print("predictions:", predictions)
print(predictions[0])
print(np.argmax(predictions[0]))
print(test_labels[0])


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, ) #cmap=plt.cm.binary

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(7), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('green')



# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()


test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)


