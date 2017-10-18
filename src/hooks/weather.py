import urllib, json
from urllib import parse, request

def makeWebhookResult(data):
    query = data.get('query')
    if query is None:
        return {}

    result = query.get('results')
    if result is None:
        return {}

    channel = result.get('channel')
    if channel is None:
        return {}

    item = channel.get('item')
    location = channel.get('location')
    units = channel.get('units')
    if (location is None) or (item is None) or (units is None):
        return {}

    condition = item.get('condition')
    if condition is None:
        return {}

    # print(json.dumps(item, indent=4))

    speech = "Today in " + location.get('city') + ": " + condition.get('text') + \
             ", the temperature is " + condition.get('temp') + " " + units.get('temperature')

    print("Response:")
    print(speech)

    return {
        "speech": speech,
        "displayText": speech,
        # "data": data,
        # "contextOut": [],
        "source": "apiai-weather-webhook-sample"
    }


def getWeather(city, date=None):
    baseurl = "https://query.yahooapis.com/v1/public/yql?"
    if date is None:
        date = 'Something'
    else:
        print('date is ')

    yqlQuery = 'select * from weather.forecast where woeid in (select woeid from geo.places(1) where text="' + city +'")'
    
    yqlUrl = baseurl + parse.urlencode({ 'q': yqlQuery }) + '&format=json'
    response = request.urlopen(yqlUrl).read()
    jsonRes = json.loads(response.decode('utf8'))

    return makeWebhookResult(jsonRes)

# print(getWeather('mangalore'))