# WeatherSense AI

WeatherSense AI is a Flask web app that pulls live weather data from OpenWeatherMap and renders a clean dashboard with condition highlights, confidence display, and city autocomplete.

## Features
- Live weather lookup by city (OpenWeatherMap current weather API)
- City autocomplete via OpenWeather geocoding API
- Visual dashboard with condition chips, stats grid, and responsive layout
- Optional ML training script to build a Random Forest model from CSV data

## Project Structure
```
weather_project/
  app.py
  train_model.py
  requirements.txt
  static/
    style.css
  templates/
    index.html
seattle-weather.csv
weatherHistory.csv
```

## Setup
1. Create a virtual environment and install dependencies.
```
pip install -r weather_project/requirements.txt
```
2. Create a .env file in the project root with your API key:
```
OPENWEATHER_API_KEY=your_api_key_here
```

## Run the App
From the workspace root:
```
python weather_project/app.py
```
Then open http://127.0.0.1:5000

## Train the ML Model (Optional)
The training script standardizes columns and trains a Random Forest model using:
`temp`, `humidity`, `pressure`, `wind_speed`, `cloud_cover`.

Run:
```
python weather_project/train_model.py
```
Artifacts saved in `weather_project/`:
- model.pkl
- scaler.pkl
- label_encoder.pkl
- static/confusion_matrix.png

## Notes
- The live dashboard currently uses OpenWeatherMap condition labels for the UI prediction.
- If the API key is missing or the city is invalid, the UI falls back to a safe empty-state.

## Requirements
See weather_project/requirements.txt for pinned versions.
