import os
import sys
import cv2
import json
import tarfile
import numpy as np
from six.moves import urllib
from keras.models import load_model
from sklearn.metrics import precision_recall_fscore_support

DATA_URL = 'http://????.tgz'
MODEL_DIR = 'multi_classes_single_label'

def init():
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        maybe_download_and_extract()
    model = load_model("./multi_classes_single_label/model_sf_vgg75.h5") # modify the address later it later
    with open("./multi_classes_single_label/multi_classes_single_label.json", 'r') as fp:
        Klass_names = json.load(fp)
    return model, Klass_names

def read_image(img_path):
    img = cv2.imread(str(img_path))
    ROW_DIM, COL_DIM = 224, 224
    ROW_DIM_ORG, COL_DIM_ORG, CHANEL_ORG = img.shape
    img_size = [ROW_DIM_ORG, COL_DIM_ORG, CHANEL_ORG]
    img = cv2.resize(img, (ROW_DIM, COL_DIM))
    return img, img_size

def predict(model, Klass_names, image_path):
    img, img_size = read_image(image_path)
    pred_class = model.predict_classes(np.expand_dims(img, axis=0))
    pred_prob = np.asarray(model.predict_proba(np.expand_dims(img, axis=0)))
    prediction_result = str(Klass_names[str(pred_class[0])])
    prediction_probability = str(int(np.max(pred_prob, axis=1) * 100))

    # print(prediction_result, prediction_probability +"%")
    # cv2.imshow(prediction_result + prediction_probability, cv2.resize(img, (img_size[1], img_size[0])))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

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
    image_path = "C:/Users/srsha/Desktop/screwdriver.jpg"
    pr_results, pr_pr = predict(MODEL, KLASS_NAMES, image_path)

