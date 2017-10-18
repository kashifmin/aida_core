from src.models import ConversationalModel
from src.hooks.weather import *
from flask import Flask, request, jsonify
import json

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

if (__name__ == "__main__"):
    app.run(debug=True, port = 5000)