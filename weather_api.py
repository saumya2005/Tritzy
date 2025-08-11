import requests

def get_weather(city_name, api_key):
    """
    Fetch current weather data from OpenWeather API for the given city.
    Returns a dictionary with weather parameters or None if error.
    """
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {
        'q': city_name,
        'appid': api_key,
        'units': 'metric'  # Get temperature in Celsius
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        weather_data = {
            'temperature': data['main']['temp'],
            'humidity': data['main']['humidity'],

        }

        return weather_data

    except requests.exceptions.RequestException as e:
        print(f"Error fetching weather data: {e}")
        return None


# Example usage:
if __name__ == "__main__":
    API_KEY = "b993eadcc639b4ce0630be182f1eecb5"
    city = input("Enter the city")  # You can change or make this input dynamic
    weather = get_weather(city, API_KEY)
    if weather:
        print(weather)
    else:
        print("Failed to get weather data.")
