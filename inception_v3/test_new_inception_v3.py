from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import cv2
import tarfile
import numpy as np
from six.moves import urllib


import argparse
import os.path
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf


DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
MODEL_DIR = 'inception-2015-12-05'


def load(label_lookup_path, uid_lookup_path):
    """Loads a human readable English name for each softmax node.
    Args:
        label_lookup_path: string UID to integer node ID.
        uid_lookup_path: string UID to human-readable string.
    Returns:
        dict from integer node ID to human-readable string.
    """
    # Loads mapping from string UID to human-readable string
    proto_as_ascii_lines = tf.gfile.GFile(uid_lookup_path).readlines()
    uid_to_human = {}
    p = re.compile(r'[n\d]*[ \S,]*')  # match digits, match everything except whitespace
    for line in proto_as_ascii_lines:
        parsed_items = p.findall(line)
        uid = parsed_items[0]
        human_string = parsed_items[2]
        uid_to_human[uid] = human_string

    # Loads mapping from string UID to integer node ID.
    node_id_to_uid = {}
    proto_as_ascii = tf.gfile.GFile(label_lookup_path).readlines()
    for line in proto_as_ascii:
        if line.startswith('  target_class:'):
            target_class = int(line.split(': ')[1])
        if line.startswith('  target_class_string:'):
            target_class_string = line.split(': ')[1]
            node_id_to_uid[target_class] = target_class_string[1:-2]

    # Loads the final mapping of integer node ID to human-readable string
    node_id_to_name = {}
    for key, val in node_id_to_uid.items():
      if val not in uid_to_human:
        tf.logging.fatal('Failed to locate: %s', val)
      name = uid_to_human[val]
      node_id_to_name[key] = name

    return node_id_to_name

def id_to_string(self, node_id):
    if node_id not in self.node_lookup:
        return ''
    return self.node_lookup[node_id]

def create_graph():
    """Creates a graph from saved GraphDef file and returns a saver."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile(os.path.join(MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
        MODEL = graph_def
        return MODEL

def predict(model, KLASS_NAMES, image_path):
    """Runs inference on an image.
    Args:
        model: this the inception v3 graph but indeed it has no effect.
        KLASS_NAMES: names of classes.
        image: Image file path name.
    Returns:
        Two lists of strings, one for the Prediction results and the other one for the corresponding probabilities.
    """
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data},)
        predictions = np.squeeze(predictions)

    predictions_results = []
    predictions_probability =[]

    node_id_to_name = KLASS_NAMES

    top_k = predictions.argsort()[-5:][::-1]
    for node_id in top_k:
        human_string = node_id_to_name[node_id]
        score = str(int(predictions[node_id]*100))
        predictions_results.append(human_string)
        predictions_probability.append(score)
    return (predictions_results, predictions_probability)


def init():
    # MODEL, Klass_names = create_graph()
    MODEL = create_graph()
    KLASS_NAMES = load(os.path.join(MODEL_DIR, 'imagenet_2012_challenge_label_map_proto.pbtxt'),
                       os.path.join(MODEL_DIR, 'imagenet_synset_to_human_label_map.txt'))
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        maybe_download_and_extract()
    return MODEL, KLASS_NAMES

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
    # MODEL, KLASS_NAMES = init()
    # pr_results, pr_pr = predict(MODEL, KLASS_NAMES, image_path)
    MODEL, KLASS_NAMES = init()
    image_path = "C:/Users/srsha/Desktop/screwdriver.jpg"
    pr_results, pr_pr = predict(MODEL, KLASS_NAMES, image_path)
    print(pr_results, pr_pr)
