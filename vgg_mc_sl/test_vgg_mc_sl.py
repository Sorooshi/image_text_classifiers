import glob
import json
import numpy as np
from collections import  OrderedDict
from sklearn.metrics import accuracy_score, precision_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import vgg_multi_class_single_label_classifier as vgg_mc_sl

def eval(labels):
    """
    :param labels: a list of labels
    :return: a numerical list of labels
    """
    y = []
    for label in labels:
        if label == "doctor":
            y.append(1)
        elif label == 'dentist':
            y.append(2)
        elif label == 'nurse':
            y.append(3)
        elif label == 'hammer':
            y.append(4)
        elif label == 'pliers':
            y.append(5)
        elif label == 'drill':
            y.append(6)
        elif label == 'screwdriver':
            y.append(7)
        else:
            y.append(0)
    return y

image_pathes = []
labels = []

dir_paths = "test_data_2classes/[a-zA-Z]*"
directories = glob.glob(dir_paths)

for directory in directories:
    file_paths = glob.glob(directory+"/*.jpg")
    for i in range(len(file_paths)):
        image_pathes.append(file_paths[i])
        labels.append(file_paths[i].split("\\")[-2])

y_true = eval(labels)
print("y_true:", y_true, len(y_true))
print("labels:", labels, len(labels))


if __name__ == '__main__':
    after_threshold_evaluation = OrderedDict([])
    thresholds = np.arange(40, 105, 5)
    y_pred = []

    MODEL, KLASS_NAMES = vgg_mc_sl.init()
    for threshold in thresholds:
        classifier_output = []
        for i in range(len(image_pathes)):
            pr_result, pr_pr = vgg_mc_sl.predict(MODEL, KLASS_NAMES, image_pathes[i])
            if int(pr_pr) >= threshold:
                classifier_output.append(pr_result)
            else:
                classifier_output.append('other')

        y_pred = eval(classifier_output)
        print("threshold:", threshold,)
        print("y_true:", y_true)
        print("y_pred:", y_pred)

        # returns weighted average f-score = 2*(Precision * Recall)/(precision + Recall).
        # Where Precision = (Tp/(Tp+Fp)) and Recall=(Tp/(Tp + Fn))
        # f_precision[threshold], f_precision[threshold], f_score[threshold] = precision_recall_fscore_support(y_true, y_pred,  warn_for='f-score')
        after_threshold_evaluation[threshold] = list(precision_recall_fscore_support(y_true, y_pred, average='weighted'))
        # precision, accuracy, f-score


    print("after_threshold_f-score:", after_threshold_evaluation)
    precision = []
    accuracy = []
    f_score = []
    for k, v in after_threshold_evaluation.items():
        precision.append(v[0])
        accuracy.append(v[1])
        f_score.append(v[2])

    print("precision:", precision)
    print("accuracy;", accuracy)
    print("f_score:", f_score)


    fig, ax = plt.subplots()
    ax.plot(thresholds, precision, 'g--', label="precision", )
    ax.plot(thresholds, accuracy, 'k--', label="accuracy", )
    ax.plot(thresholds, f_score, 'b-+', label="F-score", )
    plt.xlabel("Threshold Value")
    ax.legend(loc='center left', shadow=True, fontsize='medium')
    plt.show()


# print(y_pred)

        # returns the fraction of correctly classified samples. [(Tp+Tn)/(Total number of samples)]
        # after_threshold_accuracy[threshold] = accuracy_score(y_true, y_pred,  normalize=True)

        # returns weighted average of the precision of each class for the multi-class task.[Tp /(Tp + Fp)]
        # after_threshold_precision[threshold] = precision_score(y_true, y_pred, average='weighted')

