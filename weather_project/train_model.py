import os
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Keep the exact required feature order for model training and prediction.
FEATURE_COLUMNS = ['temp', 'humidity', 'pressure', 'wind_speed', 'cloud_cover']
TARGET_COLUMN = 'weather_condition'


def load_dataset() -> pd.DataFrame:
    """Load dataset from local project folder or parent workspace."""
    candidate_paths = [
        os.path.join(os.path.dirname(__file__), 'seattle-weather.csv'),
        os.path.join(os.path.dirname(__file__), '..', 'seattle-weather.csv'),
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            print(f"Loading dataset from: {os.path.abspath(path)}")
            return pd.read_csv(path)

    raise FileNotFoundError(
        'Dataset not found. Place seattle-weather.csv in weather_project/ or workspace root.'
    )


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map available source columns into required model schema."""
    # Normalize input column names for flexible mapping.
    raw = df.copy()
    raw.columns = [str(col).strip().lower() for col in raw.columns]

    # Build temperature column from the best available source.
    if 'temp' in raw.columns:
        temp = raw['temp']
    elif {'temp_max', 'temp_min'}.issubset(raw.columns):
        temp = (raw['temp_max'] + raw['temp_min']) / 2.0
    elif 'temp_max' in raw.columns:
        temp = raw['temp_max']
    else:
        raise ValueError('No suitable temperature columns found for mapping to temp.')

    # Use native humidity if present, else derive a proxy from precipitation/temperature.
    if 'humidity' in raw.columns:
        humidity = raw['humidity']
    else:
        precip = raw.get('precipitation', pd.Series(np.zeros(len(raw))))
        humidity = (55 + precip * 3.5 - (temp - temp.mean()) * 1.2).clip(15, 100)

    # Use native pressure if present, else derive a realistic synthetic pressure signal.
    if 'pressure' in raw.columns:
        pressure = raw['pressure']
    else:
        wind = raw.get('wind', raw.get('wind_speed', pd.Series(np.zeros(len(raw)))))
        pressure = (1013 - wind * 1.8 - (temp - 12) * 0.7).clip(950, 1050)

    # Map wind into required wind_speed feature.
    if 'wind_speed' in raw.columns:
        wind_speed = raw['wind_speed']
    elif 'wind' in raw.columns:
        wind_speed = raw['wind']
    else:
        raise ValueError('No suitable wind column found for mapping to wind_speed.')

    # Map cloud cover directly if present, else derive from precipitation and weather labels.
    if 'cloud_cover' in raw.columns:
        cloud_cover = raw['cloud_cover']
    else:
        precip = raw.get('precipitation', pd.Series(np.zeros(len(raw))))
        weather_col = raw.get('weather', pd.Series(['clear'] * len(raw))).astype(str).str.lower()
        cloud_cover = (precip * 7.5).clip(0, 100)
        cloud_cover = np.where(weather_col.isin(['fog', 'mist', 'clouds', 'haze']), np.maximum(cloud_cover, 75), cloud_cover)
        cloud_cover = np.where(weather_col.isin(['sun', 'clear']), np.minimum(cloud_cover, 25), cloud_cover)
        cloud_cover = pd.Series(cloud_cover)

    # Locate target weather column in the source dataset.
    if TARGET_COLUMN in raw.columns:
        target = raw[TARGET_COLUMN]
    elif 'weather' in raw.columns:
        target = raw['weather']
    else:
        raise ValueError('No suitable target column found for weather_condition mapping.')

    # Map source labels to the requested standardized multi-class labels.
    target = target.astype(str).str.strip().str.lower()
    target_map = {
        'clear': 'Clear',
        'sun': 'Clear',
        'clouds': 'Clouds',
        'cloudy': 'Clouds',
        'overcast': 'Clouds',
        'fog': 'Clouds',
        'mist': 'Clouds',
        'haze': 'Clouds',
        'rain': 'Rain',
        'drizzle': 'Drizzle',
        'thunderstorm': 'Thunderstorm',
        'storm': 'Thunderstorm',
        'snow': 'Snow',
    }
    weather_condition = target.map(target_map).fillna(target.str.title())

    # Build the final canonical dataframe with exact column names.
    standardized = pd.DataFrame(
        {
            'temp': pd.to_numeric(temp, errors='coerce'),
            'humidity': pd.to_numeric(humidity, errors='coerce'),
            'pressure': pd.to_numeric(pressure, errors='coerce'),
            'wind_speed': pd.to_numeric(wind_speed, errors='coerce'),
            'cloud_cover': pd.to_numeric(cloud_cover, errors='coerce'),
            'weather_condition': weather_condition,
        }
    )

    # Drop rows with null values in required columns.
    standardized = standardized.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).reset_index(drop=True)
    return standardized


if __name__ == '__main__':
    # Step 1: Load raw dataset and show shape/sample preview.
    raw_df = load_dataset()
    print(f"Dataset shape: {raw_df.shape}")
    print('First 5 rows:')
    print(raw_df.head())

    # Step 2: Show source target value counts before mapping.
    source_target = 'weather_condition' if 'weather_condition' in raw_df.columns else 'weather'
    if source_target in raw_df.columns:
        print(f"\nValue counts for source target column '{source_target}':")
        print(raw_df[source_target].astype(str).value_counts())

    # Step 3: Standardize schema to required training format.
    df = standardize_columns(raw_df)
    print('\nValue counts for mapped target column weather_condition:')
    print(df[TARGET_COLUMN].value_counts())

    # Step 4: Encode target labels and save label encoder.
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET_COLUMN])
    joblib.dump(label_encoder, os.path.join(os.path.dirname(__file__), 'label_encoder.pkl'))

    # Step 5: Split features/labels with exact required feature order.
    X = df[FEATURE_COLUMNS].copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Step 6: Fit scaler on training features only, then persist scaler.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, os.path.join(os.path.dirname(__file__), 'scaler.pkl'))

    # Step 7: Train Random Forest classifier.
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Step 8: Evaluate model with accuracy and classification report.
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")

    class_names = list(label_encoder.classes_)
    print('\nClassification Report:')
    print(
        classification_report(
            y_test,
            y_pred,
            labels=np.arange(len(class_names)),
            target_names=class_names,
            zero_division=0,
        )
    )

    # Step 9: Plot and save confusion matrix into static assets.
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_names)))
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title('Weather Condition Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    os.makedirs(static_dir, exist_ok=True)
    plt.savefig(os.path.join(static_dir, 'confusion_matrix.png'), dpi=200)
    plt.close()

    # Step 10: Save trained model for Flask inference.
    joblib.dump(model, os.path.join(os.path.dirname(__file__), 'model.pkl'))

    # Final required summary prints.
    print("✅ Model saved! Features used: ['temp', 'humidity', 'pressure', 'wind_speed', 'cloud_cover']")
    print(f"✅ Weather condition classes: {list(label_encoder.classes_)}")
