from .object_recognition.model_utils2 import loadModel, predict
from io import BytesIO
from PIL import Image
import numpy as np 

import requests

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

class ImageClassificationModel:
    def __init__(self):
        loadModel()

    def predict(self, imageUri):
        # predictions = self.model.predict()
        try:
            # img = Image.open(BytesIO(urllib.request.urlopen(imageUri).read()))
            img = Image.open(BytesIO(requests.get(imageUri).content)).convert('RGB')
            img = load_image_into_numpy_array(img)
        except Exception as e:
            return { "error": True, "message": str(e) }
        pred = predict(img)
        print('predictions', pred)
        return pred

    