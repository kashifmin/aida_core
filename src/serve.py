from models.Conversational import ConversationalModel
from models.ImageClassification import ImageClassificationModel
from models.FaceRecognition import FaceRecognitionModel
from hooks.weather import *
from flask import Flask, request, jsonify,redirect
from PIL import Image
from io import BytesIO
from numpy import array
import base64,urllib

import json

app = Flask(__name__)
chatBot = ConversationalModel()
objectRecModel = ImageClassificationModel()
faceRecModel = FaceRecognitionModel()

@app.route("/") 
def hello(): 
    return "Hello World!"

# Routing
@app.route('/query', methods=['POST'])
def reply():
    # print('New request', request.form['message'])
    error = True
    response = ''
    try:
        response = chatBot.decode_dialogflow(request.form['message'])
        error = False
    except Exception as e:
    # print('Got response:' , response)
        print(e)
        response = 'Server error. Please try again.'

    return jsonify( { 'error': error, 'text':  response} )

@app.route('/hooks/weather', methods=['POST'])
def weatherHook():
    print('here')
    req = request.get_json(silent=True, force=True)
    print('Req', req)
    city = req.get('result').get('parameters').get('geo-city')
    res = getWeather(city)
    # res = {'done': True}
    print('weather result: ', res)
    return jsonify(res)
    # return 'working'

# @app.route('/hooks', methods=['POST'])
# def hooks():


@app.route('/object-detection',methods=['POST'])
def detectObjects():
    
    try:
        req = request.get_json(silent=True, force=True)
        print("Request", req)
        # imageUri = req.get('imageUri')
        encodedImage = req.get('imageData')
    except Exception as e:
        return jsonify(error=True, message=str(e))
    
    
    response = objectRecModel.predict(encodedImage=encodedImage)
    return jsonify(error=False, predictions=response)


@app.route('/face-recognition',methods=['POST'])
def recognizeFaces():
    try:
        req = request.get_json(silent=True, force=True)
        #print("Request", req)
        # imageUri = req.get('imageUri')
        encodedImage = req.get('imageData')
        image = array(Image.open(BytesIO(base64.decodestring(encodedImage.encode('utf8')))).convert('RGB'))
    except Exception as e:
        return jsonify(error=True, message=str(e))
    

    response = faceRecModel.recognizeFaces(image)
    return jsonify(error=False, faces=response)


if (__name__ == "__main__"):
    app.run(host = "0.0.0.0", port = 5000, debug=True)