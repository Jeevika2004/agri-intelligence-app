import requests

def get_weather(city):
    api_key = "3072c0ce66324f7d4fb269e6faa7fec6"  # Replace with your real API key
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    try:
        response = requests.get(url)
        data = response.json()
        if 'main' in data:
            temperature = data['main']['temp']
            humidity = data['main']['humidity']
            return temperature, humidity
        else:
            return None, None
    except:
        return None, None
