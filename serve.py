from models import ConversationalModel
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
    print('New request', request.form['message'])
    response = chatBot.decode_dialogflow(request.form['message'])
    print('Got response:' , response)
    return jsonify( { 'text':  response} )

if (__name__ == "__main__"):
    app.run(port = 5000)