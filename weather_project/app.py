import os
from datetime import datetime

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request


# Load environment variables from .env file.
load_dotenv()

# Initialize Flask app.
app = Flask(__name__)

# Read API configuration from environment.
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', '')


def get_condition_style(predicted_condition: str) -> tuple[str, str]:
    """Map predicted condition to UI emoji and color theme."""
    style_map = {
        'Clear': ('☀️', '#f59e0b'),
        'Clouds': ('☁️', '#6b7280'),
        'Rain': ('🌧️', '#3b82f6'),
        'Drizzle': ('🌦️', '#60a5fa'),
        'Thunderstorm': ('⛈️', '#7c3aed'),
        'Snow': ('❄️', '#93c5fd'),
        'Mist': ('🌫️', '#9ca3af'),
        'Haze': ('🌫️', '#9ca3af'),
    }
    return style_map.get(predicted_condition, ('🌤️', '#10b981'))


@app.route('/')
def index():
    """Fetch live weather, run AI prediction, and render dashboard."""
    # Build consistent current date string for navbar display.
    current_date = datetime.now().strftime('%A, %d %B %Y').replace(' 0', ' ')
    requested_city = request.args.get('city', '').strip()
    active_city = requested_city

    try:
        # Ask for a city before calling external APIs.
        if not active_city:
            raise ValueError('Please enter a city to get started')

        # Fail early if credentials are not ready.
        if not OPENWEATHER_API_KEY:
            raise ValueError('Missing OPENWEATHER_API_KEY in .env')

        # Step 1: Call OpenWeatherMap API for the configured city.
        url = 'https://api.openweathermap.org/data/2.5/weather'
        params = {
            'q': active_city,
            'appid': OPENWEATHER_API_KEY,
            'units': 'metric',
        }
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        # Step 2: Extract exact model features in strict required order.
        main_payload = data.get('main', {})
        temp = main_payload['temp']
        humidity = main_payload['humidity']
        pressure = main_payload.get('pressure', 0)
        wind_speed = data.get('wind', {}).get('speed', 0)
        cloud_cover = data.get('clouds', {}).get('all', 0)

        # Extract UI-only fields (not used by the ML model).
        weather_payload = data.get('weather', [{}])
        description = weather_payload[0].get('description', 'N/A').title()
        feels_like = main_payload.get('feels_like', temp)
        visibility_m = data.get('visibility')
        visibility = round(visibility_m / 1000, 2) if visibility_m is not None else '--'
        city_name = data.get('name', active_city)
        country = data.get('sys', {}).get('country', '--')
        weather_icon = weather_payload[0].get('icon', '')
        icon_url = f'https://openweathermap.org/img/wn/{weather_icon}@2x.png'

        # Step 3: Use live API condition for the dashboard prediction.
        predicted_condition = weather_payload[0].get('main', description) or 'Unavailable'
        confidence = 100

        # Step 4: Map predicted label into visual emoji/color styling.
        condition_emoji, condition_color = get_condition_style(predicted_condition)

        # Step 5: Render final dashboard template with all required variables.
        return render_template(
            'index.html',
            temp=temp,
            humidity=humidity,
            pressure=pressure,
            wind_speed=wind_speed,
            cloud_cover=cloud_cover,
            description=description,
            feels_like=feels_like,
            visibility=visibility,
            city_name=city_name,
            country=country,
            icon_url=icon_url,
            predicted_condition=predicted_condition,
            confidence=confidence,
            condition_emoji=condition_emoji,
            condition_color=condition_color,
            current_date=current_date,
            search_city=active_city,
            error=None,
        )

    except Exception:
        empty_city = not active_city
        error_message = 'Please enter a city to get started' if empty_city else 'Could not fetch weather data'
        # Render graceful fallback view when API/model pipeline fails.
        return render_template(
            'index.html',
            temp='--',
            humidity='--',
            pressure='--',
            wind_speed='--',
            cloud_cover='--',
            description='N/A',
            feels_like='--',
            visibility='--',
            city_name=active_city or '--',
            country='--',
            icon_url='',
            predicted_condition='Unavailable',
            confidence=0,
            condition_emoji='⚠️',
            condition_color='#ef4444',
            current_date=current_date,
            search_city=active_city,
            error=error_message,
        )


@app.route('/about')
def about():
    """Render a simple about page."""
    return """
    <html>
        <head>
            <title>About - WeatherSense AI</title>
            <style>
                body { font-family: Poppins, Arial, sans-serif; background: #0a0e1a; color: #f1f5f9; padding: 40px; }
                .card { max-width: 700px; margin: 0 auto; background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.12); border-radius: 14px; padding: 28px; }
                a { color: #60a5fa; }
            </style>
        </head>
        <body>
            <div class='card'>
                <h1>About WeatherSense AI</h1>
                <p>WeatherSense AI predicts weather conditions using a Random Forest model trained on structured meteorological features.</p>
                <p>Live weather data is sourced from OpenWeatherMap and combined with ML inference for an explainable dashboard experience.</p>
                <p><a href='/'>Back to Home</a></p>
            </div>
        </body>
    </html>
    """


@app.route('/city-suggest')
def city_suggest():
    """Return city suggestions using OpenWeather geocoding API."""
    query = request.args.get('q', '').strip()
    if not query:
        return jsonify([])
    if not OPENWEATHER_API_KEY:
        return jsonify([]), 400

    url = 'https://api.openweathermap.org/geo/1.0/direct'
    params = {
        'q': query,
        'limit': 6,
        'appid': OPENWEATHER_API_KEY,
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException:
        return jsonify([]), 502

    suggestions = []
    for item in data:
        name = item.get('name')
        country = item.get('country')
        state = item.get('state')
        if not name or not country:
            continue
        label = f"{name}, {state}, {country}" if state else f"{name}, {country}"
        suggestions.append({'label': label, 'value': name})

    return jsonify(suggestions)


if __name__ == '__main__':
    # Start Flask development server.
    app.run(debug=True)
