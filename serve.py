from src.models import ConversationalModel
from src.hooks.weather import *
from flask import Flask, request, jsonify,redirect,json
from PIL import Image
from io import BytesIO
from numpy import array
import base64,urllib


app = Flask(__name__)
chatBot = ConversationalModel()

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

@app.route('/object',methods=['POST'])
def detectObject():
    req = request.get_json(silent=True, force=True)
    print(req)
    imageUri=req.get('image').get('imageUri')
    imageData=req.get('image').get('imageData')
    
    img_arr=""
    if imageUri is None:
        img_arr=convertArrayData(imageData)
    else:
        img_arr=convertArrayUri(imageUri)
    #print(img_arr)

    try:
        response=[{
            'description': 'car',
            'score': 0.8114 },{
            'description': 'bike',
            'score': 0.9114 }]
            
    except Exception as e:
        pass
    return jsonify(labelAnnotation=response)



def convertArrayData(encoded_image):
    img=Image.open(BytesIO(base64.b64decode(encoded_image)))
    img_arr=array(img)
    return img_arr

def convertArrayUri(imageUri):
    img=Image.open(BytesIO(urllib.request.urlopen(imageUri).read()))
    img_arr=array(img)
    return img_arr

if (__name__ == "__main__"):
    app.run(debug=True, port = 5000)