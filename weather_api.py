import requests
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenWeather API configuration
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', 'demo_key')

# Track coordinates for weather fetching
TRACK_COORDINATES = {
    'Bahrain': (26.0325, 50.5106),
    'Monza': (45.6156, 9.2811),
    'Silverstone': (52.0786, -1.0169),
    'Abu Dhabi': (24.4672, 54.6031),
    'Spa': (50.4372, 5.9714),
    'Spanish Grand Prix': (41.5700, 2.2611),  # Barcelona
    'Monaco': (43.7347, 7.4206),
    'Imola': (44.3439, 11.7167),
    'Miami': (25.9581, -80.2389),
    'Suzuka': (34.8431, 136.5414),
    'Interlagos': (-23.7036, -46.6997),
    'Las Vegas': (36.1162, -115.1739),
    'Jeddah': (21.6319, 39.1044),
    'Melbourne': (-37.8497, 144.9681),
    'Baku': (40.3725, 49.8533),
    'Hungaroring': (47.5789, 19.2486),
    'Zandvoort': (52.3888, 4.5409),
    'Singapore': (1.2914, 103.8640),
    'Austin': (30.1328, -97.6411),
    'Mexico City': (19.4042, -99.0907),
    'Qatar': (25.4901, 51.4564)
}

def get_weather_from_openweather(track, year=None, month=None, day=None):
    """
    Fetch weather data from OpenWeather API for a specific location.
    For historical data, we use current weather as approximation since 
    historical weather API requires paid subscription.
    """
    if track not in TRACK_COORDINATES:
        print(f"  No coordinates for {track}, using default weather")
        return {'track_temp': 25, 'rain_prob': 0}
    
    lat, lon = TRACK_COORDINATES[track]
    
    # Check if we have a valid API key
    if OPENWEATHER_API_KEY == 'demo_key' or not OPENWEATHER_API_KEY:
        print(f"  No OpenWeather API key found, using default weather for {track}")
        # Return reasonable defaults based on track location
        defaults = {
            'Bahrain': {'track_temp': 32, 'rain_prob': 0},
            'Abu Dhabi': {'track_temp': 30, 'rain_prob': 0},
            'Monza': {'track_temp': 24, 'rain_prob': 0.2},
            'Silverstone': {'track_temp': 18, 'rain_prob': 0.4},
            'Spa': {'track_temp': 16, 'rain_prob': 0.6},
            'Spanish Grand Prix': {'track_temp': 26, 'rain_prob': 0.1},
        }
        return defaults.get(track, {'track_temp': 25, 'rain_prob': 0.2})
    
    # Use current weather API (free tier)
    url = f"http://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            temp = data['main']['temp']
            # Check for rain in weather conditions
            rain_prob = 1 if any('rain' in condition['main'].lower() for condition in data['weather']) else 0
            print(f"  Weather from OpenWeather: {temp}°C, Rain: {'Yes' if rain_prob else 'No'}")
            return {'track_temp': temp, 'rain_prob': rain_prob}
        else:
            print(f"  OpenWeather API error: {response.status_code}")
    except Exception as e:
        print(f"  Weather API error: {e}")
    
    # Fallback to default values
    defaults = {
        'Bahrain': {'track_temp': 32, 'rain_prob': 0},
        'Abu Dhabi': {'track_temp': 30, 'rain_prob': 0},
        'Monza': {'track_temp': 24, 'rain_prob': 0.2},
        'Silverstone': {'track_temp': 18, 'rain_prob': 0.4},
        'Spa': {'track_temp': 16, 'rain_prob': 0.6},
        'Spanish Grand Prix': {'track_temp': 26, 'rain_prob': 0.1},
    }
    return defaults.get(track, {'track_temp': 25, 'rain_prob': 0.2})

def setup_openweather_api():
    """
    Instructions for setting up OpenWeather API key.
    """
    print("To use real weather data, you need an OpenWeather API key:")
    print("1. Go to https://openweathermap.org/api")
    print("2. Sign up for a free account")
    print("3. Get your API key")
    print("4. Set it as environment variable: OPENWEATHER_API_KEY=your_key_here")
    print("5. Or add it to a .env file in your project")
    print("\nFor now, using default weather values based on track location.")

if __name__ == "__main__":
    # Test the weather API
    print("Testing weather API...")
    weather = get_weather_from_openweather('Silverstone')
    print(f"Silverstone weather: {weather}")
    
    if OPENWEATHER_API_KEY == 'demo_key':
        setup_openweather_api()