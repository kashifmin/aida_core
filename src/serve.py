from models.Conversational import ConversationalModel
from models.ImageClassification import ImageClassificationModel
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

@app.route('/object-detection',methods=['POST'])
def detectObjects():
    
    try:
        req = request.get_json(silent=True, force=True)
        print("Request", req)
        imageUri = req.get('imageUri')
    except Exception as e:
        return jsonify(error=True, message=str(e))
    
    
    response = objectRecModel.predict(imageUri)
    return jsonify(error=False, predictions=response)


if (__name__ == "__main__"):
    app.run(debug=True, port = 5000)