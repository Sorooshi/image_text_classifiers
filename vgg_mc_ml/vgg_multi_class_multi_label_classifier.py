import os
import sys
import cv2
import tarfile
import numpy as np
from six.moves import urllib
from keras.models import load_model


DATA_URL = 'http://????.tgz'
MODEL_DIR = 'multi_classes_multi_labels'

def init():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        maybe_download_and_extract()
    model = load_model(".\multi_classes_multi_labels\model_mL_vgg90.h5") # modify it later
    Klass_names = np.load(".\multi_classes_multi_labels\multi_classes_multi_labels.npy") # modify it later
    return model, Klass_names

def read_image(img_path):
    img = cv2.imread(str(img_path))
    print("image_path:", img_path)
    ROW_DIM, COL_DIM = 224, 224
    ROW_DIM_ORG, COL_DIM_ORG, CHANEL_ORG = img.shape
    img_size = [ROW_DIM_ORG, COL_DIM_ORG, CHANEL_ORG]
    img = cv2.resize(img, (ROW_DIM, COL_DIM))
    return img, img_size

def predict(model, Klass_names, image_path):
    img, img_size = read_image(image_path)
    classes_prob = model.predict(np.expand_dims(img, axis=0))
    index = np.argsort(classes_prob[::-1])
    prediction_result = str(Klass_names[index[0][-1]]) + \
                        ", " + str(Klass_names[index[0][-2]])

    prediction_probability = str(round(float(classes_prob[0][index[0][-1]] * 100), 1)) + "% " + \
                             str(round(float(classes_prob[0][index[0][-2]]) * 100, 1)) + "%"

    print(prediction_result, prediction_probability)

    cv2.imshow(prediction_result+prediction_probability, cv2.resize(img, (img_size[1], img_size[0])))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return prediction_result, prediction_probability


def maybe_download_and_extract():
  """Download and extract model tar file."""
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(MODEL_DIR, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)



if __name__ == '__main__':
    MODEL, KLASS_NAMES = init()
    print('KLASS_NAMES:', KLASS_NAMES)
    image_path = "C:/Users/srsha/Desktop/dress.jpg"
    pr_results, pr_pr = predict(MODEL, KLASS_NAMES, image_path)
    print(pr_results, pr_pr)

