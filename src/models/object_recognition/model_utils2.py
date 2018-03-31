import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import visualization_utils as vis_util
import cv2

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


# ## Object detection imports
# Here are the imports from the object detection module.

# In[3]:

from object_detection.utils import label_map_util

from object_detection.utils import visualization_utils as vis_util

# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[4]:

# What model to download.
BASE_FOLDER = '/usr/src/app/'
MODEL_NAME = BASE_FOLDER + 'coco_inception'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(BASE_FOLDER + 'data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# ## Download Model

# In[5]:

# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)


# ## Load a (frozen) Tensorflow model into memory.

# In[6]:



# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[7]:

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

def loadModel():
    global detection_graph, sess
    
    print("Loading model ", MODEL_NAME, " ...")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    print("Model loaded.")
    sess = tf.Session(graph=detection_graph)
    print("Session started.")

def processResult(img, res, ntop=5):
    ''' 
        Takes top 5 classes and returns their class labels, bounding box, and scores 
    '''
    height, width = img.shape[1], img.shape[2]
    predictions = []
    for i in range(ntop):
        pred = dict(categories[int(res[2][0][i])])
        pred['score'] = round(res[1][0][i] * 100, 2)
        boundingBoxRaw = list(res[0][0][i])

        # normalize coordinates wrt image size
        pred['boundingBox'] = {
            'ymin': boundingBoxRaw[0]*height,
            'xmin': boundingBoxRaw[1]*width,
            'ymax': boundingBoxRaw[2]*height,
            'xmax': boundingBoxRaw[3]*width
        }

        predictions.append(pred)
    return predictions

def processResultNew(img, classes, scores, boxes, ntop=5):
    ''' 
        Takes top 5 classes and returns their class labels, bounding box, and scores 
    '''
    height, width = img.shape[1], img.shape[2]
    predictions = []
    threshold = 30 # minimum accuracy expected
    for i in range(ntop):
        
        pred = category_index[classes[i]]
        pred['score'] = round(scores[i] * 100, 2)
        if pred['score'] < threshold:
            break
        boundingBoxRaw = list(boxes[i])

        # normalize coordinates wrt image size
        pred['boundingBox'] = {
            'ymin': boundingBoxRaw[0]*height,
            'xmin': boundingBoxRaw[1]*width,
            'ymax': boundingBoxRaw[2]*height,
            'xmax': boundingBoxRaw[3]*width
        }

        predictions.append(pred)
    return predictions


def predict(img_data):
    global detection_graph,sess
    
    image_np_expanded = np.expand_dims(img_data, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    # Actual detection.
#   (boxes, scores, classes, num_detections) = sess.run(
#       [boxes, scores, classes, num_detections],
#       feed_dict={image_tensor: image_np_expanded})
    results = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})
    (boxes, scores, classes, num_detections) = results
    print('image_data', img_data)
    # Visualization of the results of a detection.
    squeezed_boxes = np.squeeze(boxes)
    squeezed_classes = np.squeeze(classes).astype(np.int32)
    squeezed_scores = np.squeeze(scores)

    vis_util.visualize_boxes_and_labels_on_image_array(
        img_data,
        squeezed_boxes,
        squeezed_classes,
        squeezed_scores,
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)

    print(squeezed_boxes.shape, squeezed_classes.shape, squeezed_scores.shape)
    print(category_index[squeezed_classes[0]])
    print(category_index)

    cv2.imwrite('lastpred.jpg', img_data)
        
    # return processResult(image_np_expanded, results)
    return processResultNew(image_np_expanded, squeezed_classes, squeezed_scores, squeezed_boxes)