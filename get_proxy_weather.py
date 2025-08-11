# get_proxy_weather.py

def adjust_weather_with_proxies(weather_data, proxy_inputs):
    """
    Adjusts city-level weather data using proxy indicators to better reflect microclimate.
    
    Args:
        weather_data (dict): Real weather data from OpenWeatherAPI
        proxy_inputs (dict): User-selected proxies (e.g., shade, irrigation)

    Returns:
        dict: Adjusted weather data
    """

    adjusted = weather_data.copy()

    # Example adjustment rules
    shade = proxy_inputs.get("shade_level", "Medium")
    irrigation = proxy_inputs.get("irrigation", "No")

    if shade == "High":
        adjusted["temperature"] -= 2
    elif shade == "Low":
        adjusted["temperature"] += 1

    if irrigation == "Yes":
        adjusted["humidity"] += 7
    else:
        adjusted["humidity"] -= 3

    # Clip values to safe ranges if needed
    adjusted["temperature"] = max(0, adjusted["temperature"])
    adjusted["humidity"] = max(0, min(100, adjusted["humidity"]))

    return adjusted
