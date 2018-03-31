from .object_recognition.model_utils2 import loadModel, predict
from io import BytesIO
from PIL import Image
import numpy as np 
import base64
import requests

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

class ImageClassificationModel:
    def __init__(self):
        loadModel()

    def predict(self, imageUri=None, encodedImage=None):
        # predictions = self.model.predict()
        try:
            # img = Image.open(BytesIO(urllib.request.urlopen(imageUri).read()))
            img = self.getImageFrom(encodedImage=encodedImage)
        except Exception as e:
            print("Exception", str(e))
            return { "error": True, "message": str(e) }
        pred = predict(img)
        print('predictions', pred)
        return pred

    def getImageFrom(self, uri=None, encodedImage=None):
        if uri is None and encodedImage is None:
            return None
        elif uri is not None:
            img = Image.open(BytesIO(requests.get(uri).content)).convert('RGB')
            img = load_image_into_numpy_array(img)
            print(img.shape)
            return img
        else:
            print('converting imagedata')
            img = Image.open(BytesIO(base64.decodestring(encodedImage).encode('utf8'))).convert('RGB')
            print('concerted', img)
            img = load_image_into_numpy_array(img)
            return img

    