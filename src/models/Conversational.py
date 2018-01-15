class ConversationalModel:
    '''
        Describes a model for Chat bot.
    '''

    # possible types: dialogflow, custom
    def __init__(self, modelType='dialogflow'):
        if modelType != 'dialogflow' and modelType != 'custom':
            raise RuntimeError('Model types must be dialogflow or custom')
        self.type = modelType

    def decode_dialogflow(self, message):
        import apiai
        import json

        CLIENT_ACCESS_TOKEN = '6280c6c02f1d4052931fc3b3e210303f'
        ai = apiai.ApiAI(CLIENT_ACCESS_TOKEN)

        request = ai.text_request()
        # session id must be different for each client tbh
        request.session_id = '3'

        request.query = message

        response = request.getresponse()
        res = response.read().decode('utf8')
        print(res)
        jsonResponse = json.loads(res)

        print('JSON : ', jsonResponse)

        return jsonResponse['result']['fulfillment']['speech']

    def decode_custom(self, message):
        return 'Custom decoder not implemented :('

    def query(self, message):
        if self.type == 'dialogflow':
            return self.decode_dialogflow(message)
        else:
            return self.decode_custom(message)
