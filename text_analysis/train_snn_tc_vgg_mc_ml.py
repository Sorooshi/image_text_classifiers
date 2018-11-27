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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from keras.layers import Activation, Dense, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import text_to_word_sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger

np.set_printoptions(suppress=True, linewidth=120, )

NN_NAME = 'snn_tc_vgg_mc_ml' # create a directory to store; snn_tc stands for simple_nn_text_classifier
if not os.path.exists(NN_NAME):
    os.mkdir(NN_NAME)

# Network Structure
def create_model(vocab_size, num_klasses):
    model = Sequential()
    model.add(Dense(512, input_shape=(vocab_size,),activation='relu')) #First layer
    model.add(Dropout(0.25))

    model.add(Dense(512, input_shape=(vocab_size,), activation='relu'))  # Second layer
    model.add(Dropout(0.25))

    model.add(Dense(num_klasses, activation='softmax'),) # Final layer

    # Optimization
    sgd = SGD(lr=0.001, nesterov=True)
    adam = Adam()

    # Compile
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
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
        print("indices:", type(test_indices), test_indices[100], train_indices[100], type(train_indices))
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
TEXTS = pd.read_csv('../img_downloader/vgg_mc_ml_text.csv', encoding='cp1252') #encoding='utf-8',
nb_words = vocab_size = int(len(set(list(TEXTS['PAGE TITLE'])))*vsp) #number of words = vocabulary size.
print("vocab_size:", nb_words, vocab_size)

data_properties = TEXTS.groupby('PARENT NAME').agg(['count'])
MAX_SEQUENCE_LENGTH = data_properties.max()


X = TEXTS['PAGE TITLE'].values # text topics of the images
Y = TEXTS['PARENT NAME'].values # corresponding class label of each image topic

klass_list = queries_list = ['black dress', 'black scarf', 'black watch', 'others', 'red dress', 'red scarf', 'red watch',
                'white dress', 'white watch',]

N, K = Y.size, len(klass_list)
XX, YY = [], []
for i in range(N):
   for klass in range(K):
       if klass_list[klass] == Y[i]:
           YY.append(Y[i].split())
           XX.append(X[i])

print("XX:", type(XX), len(XX), XX[200])
print("YY:", type(YY), len(YY), YY[200])

# Train test split:
ttsp = 0.2 #A float (0,1] to determine the ratio of train and test split
# train_texts, train_labels, test_texts, test_labels = train_test_data_split(X=XX, Y=YY, test_size=ttsp)
train_texts, test_texts, train_labels, test_labels = train_test_split(XX, YY, test_size=0.2, random_state=42,)
print("training data:", len(train_texts), len(train_labels), train_labels[100])
print("testing data:", len(test_labels), len(test_labels))

# define Tokenizer with Vocab Size:
# To vectorize a text corpus, by turning each text into either a sequence of integers
# (each integer being the index of a token in a dictionary
tokenizer = Tokenizer(nb_words=nb_words)
tokenizer.fit_on_texts(train_texts)
x_train = tokenizer.texts_to_matrix(train_texts, mode='count')
x_test = tokenizer.texts_to_matrix(test_texts, mode='count')

# converting the labels into nne-hot vector format
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_labels)
# print("y_train:", y_train)
y_test = mlb.fit_transform(test_labels)
# print("y_test:", y_test)
klass_names = mlb.classes_
num_klasses = len(klass_names)
print("classes:", num_klasses, len(klass_names))
print(klass_names)
print("x train shape", x_train.shape, )
print("x test shape", x_test.shape, )
print("y train shape", y_train.shape, )
print("y test shape", y_test.shape, )


snn_tc_labels = OrderedDict([])
for i in range(num_klasses):
    snn_tc_labels[str(i)] = klass_names[i]

# save name of the classes
write_json(snn_tc_labels, path=os.path.join(NN_NAME), file_name=NN_NAME+'-labels')

# sns.countplot(TEXTS['PARENT NAME'])
# plt.xlabel('Labels')
# plt.ylabel('Labels Count')
# plt.show()
#
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
checkpoint_saver = ModelCheckpoint(filepath=file_path, monitor='val_loss', verbose=1,
                                   save_best_only=False, save_weights_only=False, mode='auto', period=4)

early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, mode='auto')
csv_logger = CSVLogger(NN_NAME + "/" +'snn_tc_training.log')

model = create_model(vocab_size=vocab_size, num_klasses=num_klasses)
history = model.fit(x_train, y_train, epochs=num_epochs, batch_size=batch_size,
                    validation_split=0.15, verbose=1, callbacks=[checkpoint_saver, early_stopping, csv_logger]) #,

'''
import itertools
from keras.models import load_model

MODEL = load_model(NN_NAME+'/'+"/snn-model-04-0.851-0.309.hdf5")

# # Evaluate the accuracy of our trained model
score = MODEL.evaluate(x_test, y_test, batch_size=32, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# some individual evaluations:
for i in (np.arange(10, len(y_test), 50)):
    print("i:", i)
    prediction = MODEL.predict(np.array([x_test[i]]))
    predicted_indices = list( itertools.chain.from_iterable(klass_names[np.argsort(prediction)]))
    # print("predicted_indices:", predicted_indices, predicted_indices[-2:])
    predicted_label = [predicted_indices[-2:]]
    print(test_texts[i])
    print('Actual label:', test_labels[i])
    print("Predicted label:", predicted_label)
    print(" ")


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

