from src.models import ConversationalModel
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

if (__name__ == "__main__"):
    app.run(port = 5000)