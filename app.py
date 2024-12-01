import os
import sys
import threading
import time
import logging
import numpy as np
import pandas as pd
from flask import Flask, render_template, jsonify, request, redirect, url_for, session, send_file
from datetime import datetime, timedelta
from functools import wraps
import json
import warnings
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from statsmodels.tsa.seasonal import seasonal_decompose
import plotly.graph_objects as go
import plotly.utils
from google.cloud import storage
from google.oauth2 import service_account
import firebase_admin
from firebase_admin import credentials, auth
import tensorflow as tf
import joblib
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from io import BytesIO

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

FIREBASE_CONFIG = {
    "apiKey": "",
    "authDomain": ".firebaseapp.com",
    "projectId": "log-analysis-4a093",
    "storageBucket": "",
    "messagingSenderId": "163835798330",
    "appId": "",
    "measurementId": "G-TVY7PBZ4E1"
}

model_cache = {}

publisher_thread = None
publisher_stop_event = threading.Event()

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = os.urandom(24)

try:
    if not firebase_admin._apps:
        cred = credentials.Certificate('service-account.json')
        firebase_admin.initialize_app(cred)
        logger.info("Firebase Admin SDK initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Firebase Admin SDK: {str(e)}")

BUCKET_NAME = os.getenv('BUCKET_NAME', 'manufacturing-sensor-data')
BLOB_NAME = 'manufacturing_sensor_data.csv'

publisher_dir = os.path.abspath('real_time_pipeline/publisher')
if publisher_dir not in sys.path:
    sys.path.append(publisher_dir)

try:
    import sensor_publisher
except ImportError as e:
    logger.error(f"Error importing sensor_publisher: {str(e)}")

def validate_data(df, min_records=50):
    """Validate data meets minimum requirements"""
    try:
        if df is None:
            logger.error("Data validation failed: DataFrame is None")
            return False

        if len(df) < min_records:
            logger.error(f"Data validation failed: Insufficient records ({len(df)} < {min_records})")
            return False

        required_columns = ['timestamp', 'machine_id', 'temperature_reading',
                            'vibration_reading', 'pressure_reading', 'rpm_reading', 'state']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            logger.error(f"Data validation failed: Missing columns {missing_columns}")
            return False

        return True
    except Exception as e:
        logger.error(f"Error in data validation: {str(e)}")
        return False

def load_model_from_gcs(machine_id):
    """Load ML model from Google Cloud Storage"""
    try:
        logger.info(f"Attempting to load model for {machine_id}")

        if machine_id in model_cache:
            logger.info(f"Using cached model for {machine_id}")
            return model_cache[machine_id]

        credentials = service_account.Credentials.from_service_account_file(
            'service-account.json',
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        storage_client = storage.Client(credentials=credentials)
        bucket = storage_client.bucket(BUCKET_NAME)

        model_path = f'models/{machine_id}'
        model_blob = bucket.blob(f'{model_path}/lstm_model.keras')
        scaler_blob = bucket.blob(f'{model_path}/scaler.pkl')

        if not model_blob.exists() or not scaler_blob.exists():
            logger.warning(f"Model files not found for {machine_id}")
            return None, None

        temp_dir = f'/tmp/model_{machine_id}'
        os.makedirs(temp_dir, exist_ok=True)

        model_blob.download_to_filename(f'{temp_dir}/model.keras')
        scaler_blob.download_to_filename(f'{temp_dir}/scaler.pkl')

        model = tf.keras.models.load_model(f'{temp_dir}/model.keras')
        scaler = joblib.load(f'{temp_dir}/scaler.pkl')

        model_cache[machine_id] = (model, scaler)
        logger.info(f"Successfully loaded model for {machine_id}")

        return model, scaler
    except Exception as e:
        logger.error(f"Error loading model for {machine_id}: {str(e)}")
        return None, None

def load_data_from_gcs():
    """Load data from Google Cloud Storage"""
    try:
        logger.info(f"Attempting to load data from GCS bucket: {BUCKET_NAME}")
        credentials = service_account.Credentials.from_service_account_file(
            'service-account.json',
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )

        storage_client = storage.Client(credentials=credentials)
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(BLOB_NAME)

        if not blob.exists():
            logger.error(f"Data file {BLOB_NAME} not found in bucket {BUCKET_NAME}")
            return None

        content = blob.download_as_string()
        df = pd.read_csv(io.BytesIO(content))
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        if not validate_data(df):
            return None

        logger.info(f"Successfully loaded {len(df)} records")
        return df
    except Exception as e:
        logger.error(f"Error loading data from GCS: {str(e)}")
        return None

def calculate_health_score(sensor_data):
    """Calculate machine health score based on sensor readings with improved validation"""
    try:
        required_readings = ['temperature_reading', 'vibration_reading',
                             'pressure_reading', 'rpm_reading']
        for reading in required_readings:
            if reading not in sensor_data:
                raise ValueError(f"Missing required sensor reading: {reading}")

        temp_score = max(0, min(25, 25 * (1 - (sensor_data['temperature_reading'] - 65) / 20)))
        vibration_score = max(0, min(25, 25 * (1 - (sensor_data['vibration_reading'] - 0.5) / 0.3)))
        pressure_score = max(0, min(25, 25 * (1 - abs(sensor_data['pressure_reading'] - 100) / 30)))
        rpm_score = max(0, min(25, 25 * (1 - abs(sensor_data['rpm_reading'] - 1000) / 200)))

        health_score = temp_score + vibration_score + pressure_score + rpm_score

        return round(health_score, 2)
    except Exception as e:
        logger.error(f"Error calculating health score: {str(e)}")
        return 50  

def predict_failures(machine_data, machine_id):
    """Enhanced failure prediction with improved error handling and validation"""
    try:
        logger.info(f"Starting failure prediction for {machine_id}")

        if not validate_data(machine_data, min_records=30):  
            logger.warning(f"Insufficient data for prediction: {machine_id}")
            return {
                'failure_probability': 0.5,
                'risk_level': 'Unknown',
                'days_until_maintenance': 30,
                'confidence': 'Low',
                'warning': 'Insufficient data for accurate prediction'
            }

        features = ['temperature_reading', 'vibration_reading',
                    'pressure_reading', 'rpm_reading']
        recent_data = machine_data[features].tail(100)

        weights = {
            'temperature_reading': 0.4,
            'vibration_reading': 0.3,
            'pressure_reading': 0.2,
            'rpm_reading': 0.1
        }

        probabilities = {}
        trends = {}

        for feature in features:
            data = recent_data[feature]
            trend = np.polyfit(range(len(data)), data, 1)[0]
            trends[feature] = trend
            current_val = data.iloc[-1]

            if feature == 'temperature_reading':
                prob = max(0, min(1, (current_val - 65) / 20))
                if current_val > 80:
                    prob *= 1.5
            elif feature == 'vibration_reading':
                prob = max(0, min(1, (current_val - 0.5) / 0.3))
                if current_val > 0.75:
                    prob *= 1.5
            elif feature == 'pressure_reading':
                prob = max(0, min(1, abs(current_val - 100) / 30))
            else:
                prob = max(0, min(1, abs(current_val - 1000) / 200))

            if trend > 0:
                prob = min(1.0, prob * (1.2 + abs(trend)))

            probabilities[feature] = prob

        failure_prob = sum(probabilities[f] * weights[f] for f in features)
        failure_prob = min(1.0, failure_prob)

        if failure_prob > 0.7:
            risk_level = 'High'
            days_until = max(1, int(5 * (1 - failure_prob)))
        elif failure_prob > 0.4:
            risk_level = 'Medium'
            days_until = max(5, int(10 * (1 - failure_prob)))
        else:
            risk_level = 'Low'
            days_until = max(10, int(30 * (1 - failure_prob)))

        confidence_score = min(1.0, len(machine_data) / 1000)
        confidence = 'High' if confidence_score > 0.8 else 'Medium' if confidence_score > 0.5 else 'Low'

        prediction_result = {
            'failure_probability': failure_prob,
            'risk_level': risk_level,
            'days_until_maintenance': days_until,
            'confidence': confidence,
            'contributing_factors': [f for f, p in probabilities.items() if p > 0.6],
            'trend_analysis': {f: {'trend': t, 'impact': 'High' if abs(t) > 0.1 else 'Low'}
                               for f, t in trends.items()}
        }

        logger.info(f"Completed failure prediction for {machine_id}")
        return prediction_result

    except Exception as e:
        logger.error(f"Error predicting failures for {machine_id}: {str(e)}")
        return None

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_token' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_token' not in session or not session.get('is_admin'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/api/time_series_analysis/<machine_id>')
@login_required
def time_series_analysis(machine_id):
    """Enhanced time-series analysis with better error handling"""
    try:
        logger.info(f"Starting time series analysis for {machine_id}")
        df = load_data_from_gcs()
        if df is None:
            logger.error("Failed to load data for time series analysis")
            return jsonify({'error': 'Could not load data'}), 500

        machine_data = df[df['machine_id'] == machine_id].copy()
        if len(machine_data) < 30:
            return jsonify({'error': 'Insufficient data for analysis'}), 400

        sensors = ['temperature_reading', 'vibration_reading', 'pressure_reading', 'rpm_reading']
        windows = [5, 15, 30]
        analysis = {}

        for sensor in sensors:
            try:
                ma_dict = {}
                for w in windows:
                    ma = machine_data[sensor].rolling(window=w, min_periods=1).mean()
                    ma_dict[f'{w}min'] = ma.tolist()

                std_dev = machine_data[sensor].rolling(window=15, min_periods=1).std()

                try:
                    filled_data = machine_data[sensor].fillna(method='ffill').fillna(method='bfill')
                    if len(filled_data) < 24:
                        period = len(filled_data) // 2
                    else:
                        period = 24

                    seasonal_result = seasonal_decompose(filled_data, period=period)
                    seasonality = seasonal_result.seasonal.fillna(0).tolist()
                    trend = seasonal_result.trend.fillna(method='ffill').fillna(method='bfill').tolist()
                except Exception as se:
                    logger.warning(f"Error in seasonal decomposition for {sensor}: {str(se)}")
                    seasonality = []
                    trend = machine_data[sensor].rolling(window=12, min_periods=1).mean().tolist()

                recent_data = machine_data[sensor].tail(50)
                slope = np.polyfit(range(len(recent_data)), recent_data, 1)[0]

                analysis[sensor] = {
                    'current_value': float(machine_data[sensor].iloc[-1]),
                    'moving_averages': ma_dict,
                    'std_deviation': std_dev.tolist(),
                    'seasonality': seasonality,
                    'trend': trend,
                    'trend_direction': 'Increasing' if slope > 0 else 'Decreasing',
                    'trend_strength': abs(slope),
                    'statistics': {
                        'mean': float(machine_data[sensor].mean()),
                        'std': float(machine_data[sensor].std()),
                        'min': float(machine_data[sensor].min()),
                        'max': float(machine_data[sensor].max())
                    }
                }

            except Exception as e:
                logger.error(f"Error processing sensor {sensor}: {str(e)}")
                analysis[sensor] = {'error': str(e)}

        try:
            fig = go.Figure()
            for sensor in sensors:
                fig.add_trace(go.Scatter(
                    x=machine_data['timestamp'],
                    y=machine_data[sensor],
                    name=sensor.replace('_reading', '').title(),
                    mode='lines'
                ))

                ma = machine_data[sensor].rolling(window=15).mean()
                fig.add_trace(go.Scatter(
                    x=machine_data['timestamp'],
                    y=ma,
                    name=f"{sensor.replace('_reading', '')} (MA)",
                    line=dict(dash='dash'),
                    opacity=0.7
                ))

            fig.update_layout(
                title=f'Sensor Readings Analysis - {machine_id}',
                xaxis_title='Time',
                yaxis_title='Values',
                height=600,
                showlegend=True,
                hovermode='x unified'
            )

            analysis['visualization'] = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            analysis['visualization'] = None

        return jsonify(analysis)
    except Exception as e:
        logger.error(f"Error in time series analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/anomaly_detection/<machine_id>')
@login_required
def anomaly_detection(machine_id):
    """Enhanced anomaly detection with improved visualization"""
    try:
        logger.info(f"Starting anomaly detection for {machine_id}")
        df = load_data_from_gcs()
        if df is None:
            return jsonify({'error': 'Could not load data'}), 500

        machine_data = df[df['machine_id'] == machine_id].copy()
        if len(machine_data) < 30:
            return jsonify({'error': 'Insufficient data for analysis'}), 400

        features = ['temperature_reading', 'vibration_reading', 'pressure_reading', 'rpm_reading']
        scaler = StandardScaler()
        X = scaler.fit_transform(machine_data[features])

        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(X)

        anomalies = machine_data[clusters == -1].copy()
        normal_data = machine_data[clusters != -1].copy()

        thresholds = {}
        for feature in features:
            mean = normal_data[feature].mean()
            std = normal_data[feature].std()
            thresholds[feature] = {
                'mean': float(mean),
                'std': float(std),
                'lower': float(mean - 2 * std),
                'upper': float(mean + 2 * std)
            }

        anomaly_details = []
        for idx, row in anomalies.iterrows():
            severity = 0
            reasons = []
            feature_analysis = {}

            for feature in features:
                value = row[feature]
                thresh = thresholds[feature]
                deviation = abs((value - thresh['mean']) / thresh['std'])

                feature_analysis[feature] = {
                    'value': float(value),
                    'deviation': float(deviation),
                    'threshold_violated': 'upper' if value > thresh['upper'] else 'lower' if value < thresh['lower'] else None
                }

                if deviation > 2:
                    severity += 1
                    reasons.append(f"{feature.replace('_reading', '')} deviation: {deviation:.2f}σ")

            anomaly_details.append({
                'timestamp': row['timestamp'].isoformat(),
                'severity': 'High' if severity > 2 else 'Medium' if severity > 1 else 'Low',
                'reasons': reasons,
                'feature_analysis': feature_analysis
            })

        try:
            fig = go.Figure()
            for feature in features:
                fig.add_trace(go.Scatter(
                    x=machine_data['timestamp'],
                    y=machine_data[feature],
                    name=feature.replace('_reading', ''),
                    mode='lines',
                    line=dict(width=1)
                ))

                if len(anomalies) > 0:
                    fig.add_trace(go.Scatter(
                        x=anomalies['timestamp'],
                        y=anomalies[feature],
                        mode='markers',
                        name=f'{feature} Anomalies',
                        marker=dict(
                            size=8,
                            symbol='x',
                            color='red'
                        )
                    ))

            fig.update_layout(
                title=f'Anomaly Detection Results - {machine_id}',
                xaxis_title='Time',
                yaxis_title='Values',
                height=600,
                showlegend=True,
                hovermode='closest'
            )

            visualization = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        except Exception as e:
            logger.error(f"Error creating visualization: {str(e)}")
            visualization = None

        return jsonify({
            'anomalies': anomaly_details,
            'total_anomalies': len(anomalies),
            'anomaly_percentage': (len(anomalies) / len(machine_data)) * 100,
            'thresholds': thresholds,
            'visualization': visualization
        })
    except Exception as e:
        logger.error(f"Error in anomaly detection: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictive_analytics/<machine_id>')
@login_required
def predictive_analytics(machine_id):
    """Enhanced predictive analytics with improved feature engineering"""
    try:
        logger.info(f"Starting predictive analytics for {machine_id}")
        df = load_data_from_gcs()
        if df is None:
            return jsonify({'error': 'Could not load data'}), 500

        machine_data = df[df['machine_id'] == machine_id].copy()
        if len(machine_data) < 30:
            return jsonify({'error': 'Insufficient data for analysis'}), 400

        machine_data['hour'] = machine_data['timestamp'].dt.hour
        machine_data['day_of_week'] = machine_data['timestamp'].dt.dayofweek
        machine_data['is_weekend'] = machine_data['day_of_week'].isin([5, 6]).astype(int)

        sensors = ['temperature_reading', 'vibration_reading', 'pressure_reading', 'rpm_reading']

        for sensor in sensors:
            machine_data[f'{sensor}_rolling_mean'] = machine_data[sensor].rolling(10).mean()
            machine_data[f'{sensor}_rolling_std'] = machine_data[sensor].rolling(10).std()
            machine_data[f'{sensor}_trend'] = machine_data[sensor].diff()

            machine_data[f'{sensor}_rate'] = machine_data[sensor].pct_change()

            if sensor == 'temperature_reading':
                threshold = 85
            elif sensor == 'vibration_reading':
                threshold = 0.8
            elif sensor == 'pressure_reading':
                threshold = 130
            else:
                threshold = 1200

            machine_data[f'{sensor}_threshold_violation'] = (machine_data[sensor] > threshold).astype(int)

        feature_cols = ['hour', 'day_of_week', 'is_weekend'] + [
            col for col in machine_data.columns if '_rolling' in col
                                                   or '_trend' in col or '_rate' in col or '_threshold_violation' in col
        ]

        features = machine_data[feature_cols].fillna(method='ffill').fillna(0)

        target = (machine_data['state'] == 'maintenance').astype(int)

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )

        split_idx = int(len(features) * 0.8)
        model.fit(features[:split_idx], target[:split_idx])

        predictions = model.predict(features[split_idx:])
        actuals = target[split_idx:]

        importance = dict(sorted(
            zip(feature_cols, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        ))

        trends = {}
        for sensor in sensors:
            recent_data = machine_data[sensor].tail(100)
            trend = np.polyfit(range(len(recent_data)), recent_data, 1)[0]
            mean = recent_data.mean()
            std = recent_data.std()

            trends[sensor] = {
                'direction': 'Increasing' if trend > 0 else 'Decreasing',
                'magnitude': abs(trend),
                'significance': 'High' if abs(trend) > std else 'Low',
                'current_value': float(recent_data.iloc[-1]),
                'mean': float(mean),
                'std': float(std),
                'stability': 'Stable' if std / mean < 0.1 else 'Unstable'
            }

        try:
            importance_fig = go.Figure([go.Bar(
                x=list(importance.values())[:10],
                y=list(importance.keys())[:10],
                orientation='h'
            )])

            importance_fig.update_layout(
                title='Top 10 Feature Importance',
                xaxis_title='Importance Score',
                yaxis_title='Feature',
                height=400
            )

            predictions_fig = go.Figure()
            predictions_fig.add_trace(go.Scatter(
                y=actuals,
                name='Actual',
                mode='lines'
            ))
            predictions_fig.add_trace(go.Scatter(
                y=predictions,
                name='Predicted',
                mode='lines'
            ))

            predictions_fig.update_layout(
                title='Maintenance Predictions vs Actuals',
                xaxis_title='Time',
                yaxis_title='Probability',
                height=400
            )

            visualizations = {
                'importance': json.dumps(importance_fig, cls=plotly.utils.PlotlyJSONEncoder),
                'predictions': json.dumps(predictions_fig, cls=plotly.utils.PlotlyJSONEncoder)
            }
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
            visualizations = None

        return jsonify({
            'model_performance': {
                'accuracy': float(model.score(features[split_idx:], target[split_idx:])),
                'predictions': predictions.tolist(),
                'actuals': actuals.tolist()
            },
            'feature_importance': importance,
            'trends': trends,
            'maintenance_prediction': {
                'next_maintenance_needed': bool(predictions[-1] > 0.5),
                'probability': float(predictions[-1]),
                'confidence': float(model.score(features[split_idx:], target[split_idx:]) * 100)
            },
            'visualizations': visualizations
        })

    except Exception as e:
        logger.error(f"Error in predictive analytics: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/machine_history/<machine_id>')
@login_required
def machine_history(machine_id):
    """API endpoint to retrieve machine history data."""
    try:
        logger.info(f"Fetching history for machine: {machine_id}")
        df = load_data_from_gcs()
        if df is None:
            return jsonify({'error': 'Could not load data from Cloud Storage'}), 500

        machine_data = df[df['machine_id'] == machine_id]
        if machine_data.empty:
            return jsonify({'error': f'No data found for machine {machine_id}'}), 404

        machine_data = machine_data.sort_values('timestamp')

        machine_data['timestamp'] = machine_data['timestamp'].dt.strftime('%Y-%m-%dT%H:%M:%S')

        state_counts = machine_data['state'].value_counts(normalize=True) * 100
        state_durations = machine_data.groupby('state')['timestamp'].count() * 1

        state_analysis = {
            'distribution': {
                state: {
                    'percentage': round(state_counts[state], 2),
                    'duration_hours': int(state_durations[state])
                } for state in state_counts.index
            }
        }

        sensors = ['temperature_reading', 'vibration_reading', 'pressure_reading', 'rpm_reading']
        trends = {}
        for sensor in sensors:
            data = machine_data[sensor]
            trend = np.polyfit(range(len(data)), data, 1)[0]
            trends[sensor] = {
                'direction': 'Increasing' if trend > 0 else 'Decreasing',
                'strength': abs(trend * 100)  
            }

        fig = go.Figure()
        for sensor in sensors:
            fig.add_trace(go.Scatter(
                x=machine_data['timestamp'],
                y=machine_data[sensor],
                name=sensor.replace('_reading', '').title(),
                mode='lines'
            ))
        fig.update_layout(
            title=f'Machine History - {machine_id}',
            xaxis_title='Time',
            yaxis_title='Sensor Values',
            height=600,
            showlegend=True
        )

        plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

        return jsonify({
            'state_analysis': state_analysis,
            'trends': trends,
            'plot': plot_json
        })
    except Exception as e:
        logger.error(f"Error fetching machine history for {machine_id}: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
@login_required
def index():
    """Main dashboard route"""
    logger.info("Accessing index route")
    return render_template('index.html',
                           firebase_config=FIREBASE_CONFIG)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login route with enhanced error handling"""
    logger.info(f"Accessing login route with method: {request.method}")
    if request.method == 'GET':
        return render_template('login.html', firebase_config=FIREBASE_CONFIG)

    if request.method == 'POST':
        try:
            token = request.json.get('idToken')
            if not token:
                logger.warning("No token provided in login request")
                return jsonify({'error': 'No token provided'}), 400

            decoded_token = auth.verify_id_token(token)
            session['user_token'] = token
            session['user_id'] = decoded_token['uid']
            session['user_email'] = decoded_token.get('email', '')
            session['is_admin'] = decoded_token.get('admin', False)

            logger.info(f"User {decoded_token['uid']} logged in successfully")
            return jsonify({'status': 'success'})
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return jsonify({'error': str(e)}), 401

@app.route('/logout')
def logout():
    """Logout route"""
    try:
        user_email = session.get('user_email', 'Unknown user')
        logger.info(f"User logging out: {user_email}")
        session.clear()
        return redirect(url_for('login'))
    except Exception as e:
        logger.error(f"Error during logout: {str(e)}")
        return redirect(url_for('login'))

@app.route('/register')
def register():
    """Registration page route"""
    logger.info("Accessing register page")
    return render_template('register.html', firebase_config=FIREBASE_CONFIG)

@app.route('/api/register', methods=['POST'])
def api_register():
    """API endpoint for user registration with enhanced validation"""
    try:
        data = request.json
        token = data.get('idToken')
        email = data.get('email')

        if not token or not email:
            logger.warning("Missing required fields for registration")
            return jsonify({'error': 'Missing required fields'}), 400

        decoded_token = auth.verify_id_token(token)
        uid = decoded_token['uid']

        custom_claims = {
            'admin': False,
            'registered_at': datetime.now().isoformat()
        }
        auth.set_custom_user_claims(uid, custom_claims)

        logger.info(f"User registered successfully: {email}")
        return jsonify({'status': 'success', 'message': 'Registration successful'})
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        return jsonify({'error': str(e)}), 400

@app.route('/ml_insights')
@login_required
def ml_insights():
    """ML insights dashboard route"""
    logger.info("Accessing ML insights page")
    try:
        df = load_data_from_gcs()
        if df is None:
            logger.error("No data available for ML insights")
            return render_template('ml_insights.html',
                                   firebase_config=FIREBASE_CONFIG,
                                   error="No data available")

        return render_template('ml_insights.html',
                               firebase_config=FIREBASE_CONFIG)
    except Exception as e:
        logger.error(f"Error loading ML insights page: {str(e)}")
        return render_template('ml_insights.html',
                               firebase_config=FIREBASE_CONFIG,
                               error="Error loading insights")

@app.route('/api/start_real_time')
@admin_required
def start_real_time():
    """Start real-time data publishing"""
    try:
        start_real_time_publishing()
        return jsonify({'status': 'success', 'message': 'Real-time publishing started'})
    except Exception as e:
        logger.error(f"Error starting real-time publishing: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop_real_time')
@admin_required
def stop_real_time():
    """Stop real-time data publishing"""
    try:
        stop_real_time_publishing()
        return jsonify({'status': 'success', 'message': 'Real-time publishing stopped'})
    except Exception as e:
        logger.error(f"Error stopping real-time publishing: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/machine_status')
@login_required
def machine_status():
    """API endpoint for machine status with enhanced error handling"""
    try:
        logger.info("Fetching machine status")
        df = load_data_from_gcs()
        if df is None:
            return jsonify({'error': 'Could not load data from Cloud Storage'}), 500

        machines = df['machine_id'].unique()
        status_data = {}

        for machine in machines:
            try:
                machine_data = df[df['machine_id'] == machine].copy()
                if len(machine_data) < 1:
                    continue

                latest_data = machine_data.iloc[-1]
                health_score = calculate_health_score(latest_data)

                ml_predictions = predict_failures(machine_data, machine)
                if ml_predictions is None:
                    ml_predictions = {
                        'failure_probability': 0.5,
                        'risk_level': 'Unknown',
                        'days_until_maintenance': 30,
                        'confidence': 'Low'
                    }

                sensor_status = {}
                for sensor, threshold in {
                    'temperature': (85, latest_data['temperature_reading']),
                    'vibration': (0.8, latest_data['vibration_reading']),
                    'pressure': (130, latest_data['pressure_reading']),
                    'rpm': (1200, latest_data['rpm_reading'])
                }.items():
                    threshold_val, current_val = threshold
                    trend = machine_data[f'{sensor}_reading'].tail(10).diff().mean()

                    sensor_status[sensor] = {
                        'current': float(current_val),
                        'status': 'Critical' if current_val > threshold_val else 'Normal',
                        'trend': 'Increasing' if trend > 0 else 'Decreasing',
                        'trend_value': float(trend)
                    }

                status_data[machine] = {
                    'current_state': latest_data['state'],
                    'predictions': {
                        'health_score': health_score,
                        'risk_level': ml_predictions['risk_level'],
                        'days_until_maintenance': ml_predictions['days_until_maintenance'],
                        'next_maintenance_date': (datetime.now() +
                                                  timedelta(
                                                      days=ml_predictions['days_until_maintenance'])).strftime(
                            '%Y-%m-%d'),
                        'confidence': ml_predictions.get('confidence', 'Medium'),
                        'sensor_status': sensor_status
                    }
                }

            except Exception as e:
                logger.error(f"Error processing machine {machine}: {str(e)}")
                continue

        if not status_data:
            return jsonify({'error': 'No valid machine data available'}), 500

        logger.info(f"Successfully processed status for {len(status_data)} machines")
        return jsonify(status_data)
    except Exception as e:
        logger.error(f"Error in machine_status route: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/generate_report', methods=['POST'])
@admin_required
def generate_report():
    """Generate comprehensive PDF report with enhanced error handling"""
    try:
        logger.info("Starting report generation")

        df = load_data_from_gcs()
        if df is None:
            logger.error("Failed to load data for report generation")
            return jsonify({'error': 'Could not load data'}), 500

        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=landscape(letter))
        elements = []
        styles = getSampleStyleSheet()

        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=1
        )

        subtitle_style = ParagraphStyle(
            'CustomSubTitle',
            parent=styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=20,
            textColor=colors.HexColor('#1a237e')
        )

        elements.append(Paragraph("Manufacturing Analytics Report", title_style))
        elements.append(Paragraph(
            f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            styles['Normal']
        ))
        elements.append(Spacer(1, 20))

        elements.append(Paragraph("System Overview", subtitle_style))

        total_machines = len(df['machine_id'].unique())
        total_records = len(df)
        data_timespan = (df['timestamp'].max() - df['timestamp'].min()).days
        healthy_machines = 0
        critical_machines = 0

        for machine_id in df['machine_id'].unique():
            machine_data = df[df['machine_id'] == machine_id].iloc[-1]
            health_score = calculate_health_score(machine_data)
            if health_score > 70:
                healthy_machines += 1
            elif health_score < 50:
                critical_machines += 1

        overview_data = [
            ['Metric', 'Value', 'Status'],
            ['Total Machines', str(total_machines), 'N/A'],
            ['Healthy Machines', str(healthy_machines),
             'Good' if healthy_machines > total_machines * 0.7 else 'Warning'],
            ['Critical Machines', str(critical_machines),
             'Good' if critical_machines < total_machines * 0.2 else 'Warning'],
            ['Total Records', str(total_records), 'N/A'],
            ['Data Timespan (days)', str(data_timespan), 'N/A'],
            ['Avg Records per Machine', f"{total_records / total_machines:.0f}", 'N/A']
        ]

        overview_table = Table(overview_data, colWidths=[200, 100, 100])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.blue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))

        elements.append(overview_table)
        elements.append(Spacer(1, 20))

        doc.build(elements)
        buffer.seek(0)

        logger.info("Report generated successfully")
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'manufacturing_report_{datetime.now().strftime("%Y%m%d_%H%M")}.pdf'
        )

    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        return jsonify({'error': f'Error generating report: {str(e)}'}), 500

def start_real_time_publishing():
    global publisher_thread, publisher_stop_event
    if publisher_thread and publisher_thread.is_alive():
        logger.info("Real-time publishing is already running.")
        return

    publisher_stop_event.clear()

    def run_publisher():
        try:
            sensor_publisher.run_publisher(publisher_stop_event)
        except Exception as e:
            logger.error(f"Error in real-time publisher thread: {str(e)}")

    publisher_thread = threading.Thread(target=run_publisher, name="PublisherThread")
    publisher_thread.start()
    logger.info("Started real-time data publishing")

def stop_real_time_publishing():
    global publisher_thread, publisher_stop_event
    if not publisher_thread:
        logger.info("Real-time publishing is not running.")
        return

    publisher_stop_event.set()
    publisher_thread.join()
    publisher_thread = None
    logger.info("Stopped real-time data publishing")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
