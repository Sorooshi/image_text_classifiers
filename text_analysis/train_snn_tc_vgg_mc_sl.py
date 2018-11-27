#-*- coding: utf-8 -*-
import os
import glob
import h5py
import json
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import OrderedDict
from keras.models import Sequential
from keras.optimizers import SGD, Adam
from sklearn.metrics import confusion_matrix
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelBinarizer
from keras.layers import Activation, Dense, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger


NN_NAME = 'snn_tc_vgg_mc_sl' # create a directory to store; snn_tc stands for simple_nn_text_classifier
if not os.path.exists(NN_NAME):
    os.mkdir(NN_NAME)


# Network Structure
def create_model(vocab_size):
    model = Sequential()
    model.add(Dense(512, input_shape=(vocab_size,),activation='relu')) #First layer
    model.add(Dropout(0.25))

    model.add(Dense(num_klasses, activation='softmax'),) # Final layer

    # Optimization
    sgd = SGD(lr=0.001, nesterov=True)
    adam = Adam()

    # Compile
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    return model

def train_test_data_split(X, Y, test_size):
    num_samples = len(X)
    if len(X) != len(Y):
        print("SOME THING SERIOUS IS WRONG!")
        assert len(X)!= len(Y)
    else:
        index_sample = range(num_samples)
        test_portion = int(num_samples*test_size)
        train_indices = np.random.choice(index_sample, (num_samples-test_portion), replace=False)
        test_indices = list(set(index_sample).symmetric_difference(train_indices))
        train_samples, train_labels = X[train_indices], Y[train_indices]
        test_samples, test_labels = X[test_indices], Y[test_indices]
    return train_samples, train_labels, test_samples, test_labels

def write_json(dict_to_write, path, file_name):
    """
    this function muct be called when the all images of one class are download to save the text in the same folder.
    """
    with open(path + "/" + file_name + ".json", "w") as fp:
        json.dump(dict_to_write, fp)
    return None

def extended_write_json(dict_to_write, path, file_name):
    """
    this function muct be called when the all images of one class are download to save the text in the same folder.
    """
    with open(path + "/" + file_name + ".json", "w+") as fp:
        json.dump(dict_to_write, fp)
    return None

# For reproducibility
np.random.seed(1237)
vsp = 1 #float greater than zero to determine the vocabulary size according to number of unique words.
TEXTS = pd.read_csv('../img_downloader/vgg_mc_sl_text.csv', encoding='cp1252') #encoding='utf-8',
nb_words = vocab_size = int(len(set(list(TEXTS['PAGE TITLE'])))*vsp) #number of words = vocabulary size.
print("vocab_size:", nb_words, vocab_size)

data_properties = TEXTS.groupby('PARENT NAME').agg(['count'])
MAX_SEQUENCE_LENGTH = data_properties.max()


X = TEXTS['PAGE TITLE'].values # text topics of the images
Y = TEXTS['PARENT NAME'].values # corresponding class label of each image topic

# Train test split:
ttsp = 0.2 #A float (0,1] to determine the ratio of train and test split
train_texts, train_labels, test_texts, test_labels = train_test_data_split(X=X, Y=Y, test_size=ttsp)

print("train data:", train_texts.shape, train_labels.shape, type(train_texts), train_texts[100], train_labels[100])
print("test data:", test_labels.shape, test_labels.shape,type(test_texts), test_texts[100], test_labels[100])

# define Tokenizer with Vocab Size:
# To vectorize a text corpus, by turning each text into either a sequence of integers
# (each integer being the index of a token in a dictionary
tokenizer = Tokenizer(nb_words=nb_words)
tokenizer.fit_on_texts(train_texts)
x_train = tokenizer.texts_to_matrix(train_texts, mode='count')
x_test = tokenizer.texts_to_matrix(test_texts, mode='count')

# converting the labels into nne-hot vector format
lb = LabelBinarizer()
y_train = lb.fit_transform(train_labels)
y_test = lb.fit_transform(test_labels)
klass_names = lb.classes_
num_klasses = len(klass_names)
print("classes:", num_klasses, len(klass_names))
print(klass_names)
print("x train shape", x_train.shape, x_train[100])
print("x test shape", x_test.shape, x_test[100])
print("y train shape", y_train.shape, y_train[100] )
print("y test shape", y_test.shape, y_train[100])


snn_tc_labels = OrderedDict([])
for i in range(num_klasses):
    snn_tc_labels[str(i)] = klass_names[i]

# save name of the classes
write_json(snn_tc_labels, path=os.path.join(NN_NAME), file_name=NN_NAME+'-labels')

# sns.countplot(TEXTS['PARENT NAME'])
# plt.xlabel('Labels')
# plt.ylabel('Labels Count')
# plt.show()

# sns.countplot(np.reshape(train_labels, newshape=-1))
# plt.xlabel('TRAIN LABELS')
# plt.ylabel('TRAIN LABELS COUNT')
# plt.show()
#
# sns.countplot(test_labels)
# plt.xlabel('TEST LABELS')
# plt.ylabel('TEST LABELS COUNT')
# plt.show()

'''
training_results_snn_tc = OrderedDict([])
testing_results_snn_tc = OrderedDict([])

# num_epochs = np.arange(5, 50, 20)
num_epochs = 50
batch_size = 32

file_path = NN_NAME + "/" + "snn-model-{epoch:02d}-{val_acc:.3f}-{val_loss:.3f}.hdf5"
checkpoint_saver = ModelCheckpoint(filepath=file_path, monitor='val_loss', verbose=0,
                                   save_best_only=False, save_weights_only=False, mode='auto', period=4)

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
csv_logger = CSVLogger(NN_NAME + "/" +'snn_tc_training.log')

model = create_model(vocab_size=vocab_size)
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size,
                    validation_split=0.1, verbose=1, callbacks=[checkpoint_saver, early_stopping, csv_logger]) #,
'''

from keras.models import load_model

MODEL = load_model(NN_NAME+'/'+"/snn-model-04-0.941-0.202.hdf5")

# # Evaluate the accuracy of our trained model
score = MODEL.evaluate(x_test, y_test, batch_size=32, verbose=1)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# some individual evaluations:
for i in (np.arange(10,1800, 50)):
    print("i:", i)
    prediction = MODEL.predict(np.array([x_test[i]]))
    predicted_label = klass_names[np.argmax(prediction)]
    print(test_texts[i])
    print('Actual label:' + test_labels[i])
    print("Predicted label: " + predicted_label + "\n")


y_softmax = MODEL.predict(x_test)

y_test_1d = []
y_pred_1d = []

for i in range(len(y_test)):
    probs = y_test[i]
    index_arr = np.nonzero(probs)
    one_hot_index = index_arr[0].item(0)
    y_test_1d.append(one_hot_index)

for i in range(0, len(y_softmax)):
    probs = y_softmax[i]
    predicted_index = np.argmax(probs)
    y_pred_1d.append(predicted_index)

def plot_confusion_matrix(cm, classes,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=10)
    plt.xlabel('Predicted label', fontsize=10)

cnf_matrix = confusion_matrix(y_test_1d, y_pred_1d)
plt.figure(figsize=(24,25))
plot_confusion_matrix(cnf_matrix, classes=klass_names, title="Confusion matrix")
plt.savefig(NN_NAME + "/" + 'conf_mat.jpg')
plt.show()
print("cnf_matrix:",)
print(cnf_matrix)
