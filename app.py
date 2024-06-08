import json
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
from datetime import datetime

app = Flask(__name__)

# Configure the upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model
model = joblib.load('final_temperature_predictor.pkl')

# Load parameters
with open('parameters.json', 'r') as json_file:
    params = json.load(json_file)

#Chuẩn hóa dữ liệu
def scale_input(features):
    scaled_input = [
        (features['precipitation'] - params['data_precipitation_min']) / (params['data_precipitation_max'] - params['data_precipitation_min']),
        (features['humidity'] - params['data_humidity_min']) / (params['data_humidity_max'] - params['data_humidity_min']),
        (features['wind_gust'] - params['data_wind_gust_min']) / (params['data_wind_gust_max'] - params['data_wind_gust_min']),
        (features['wind_speed'] - params['data_wind_speed_min']) / (params['data_wind_speed_max'] - params['data_wind_speed_min']),
        (features['cloud_cover'] - params['data_cloud_cover_min']) / (params['data_cloud_cover_max'] - params['data_cloud_cover_min']),
        (features['pressure'] - params['data_pressure_min']) / (params['data_pressuremax'] - params['data_pressure_min'])
    ]
    return scaled_input

#Tải dữ liệu từ file json để in ra màn hình 
def load_and_preprocess_weather_data():
    with open('D:/DATN/Test-UI-2/data/data_weather.json', 'r') as json_data_file:
        data_weather = json.load(json_data_file)

    for data in data_weather:
        timestamp = data['timestamp']
        # Chuyển đổi timestamp thành dạng YYYY-MM-DD HH:MM
        formatted_timestamp = datetime.strptime(timestamp, '%Y%m%dT%H%M').strftime('%Y-%m-%d %H:%M')
        data['timestamp'] = formatted_timestamp

    last_two_weather_data = data_weather[-3:]
    return last_two_weather_data

data_weather_to_display = load_and_preprocess_weather_data()

# Home route to display the HTML form
@app.route('/')
def home():
    return render_template('index.html', data_weather=data_weather_to_display)


@app.route('/input-data', methods=['GET'])
def input_page():
    return render_template('input_data.html', data_weather=data_weather_to_display)

#Hàm dự đoán nhiệt độ dựa trên giá trị nhập vào
def predict_from_input(features):
    scaled_input = scale_input(features)
    temperature = model.predict(np.array(scaled_input).reshape(1, -1))[0]
    return temperature

# Flask route to handle prediction from input data
@app.route('/predict-input-data', methods=['POST'])
def predict_from_input_data():
    try:
        features = {
            'precipitation': float(request.form['precipitation']),
            'humidity': float(request.form['humidity']),
            'wind_gust': float(request.form['wind_gust']),
            'wind_speed': float(request.form['wind_speed']),
            'cloud_cover': float(request.form['cloud_cover']),
            'pressure': float(request.form['pressure'])
        }
        temperature = predict_from_input(features)
        y_pred_xgb_orig = (temperature * (params['data_temperature_max'] - params['data_temperature_min'])) + params['data_temperature_min']
        return render_template('input_data.html', prediction_text=f'Dự đoán nhiệt độ: {y_pred_xgb_orig:.2f}°C', data_weather = data_weather_to_display)
    except ValueError:
        return render_template('input_data.html', prediction_text="Vui lòng nhập các giá trị số hợp lệ.", data_weather = data_weather_to_display)

#Dự đoán theo tệp .xlsx hoặc .csv
@app.route('/file', methods=['GET'])
def file_page():
    return render_template('file.html', data_weather=data_weather_to_display)

def predict_from_file(file):
    df = pd.read_excel(file) if file.filename.endswith('.xlsx') else pd.read_csv(file)
    predictions = []
    for _, row in df.iterrows():
        features = {
            'precipitation': row['Precipitation Total'],
            'humidity': row['Relative Humidity [2 m]'],
            'wind_gust': row['Wind Gust'],
            'wind_speed': row['Wind Speed [100 m]'],
            'cloud_cover': row['Cloud Cover Total'],
            'pressure': row['Mean Sea Level Pressure [MSL]']
        }
        formatted_timestamp = datetime.strptime(row['timestamp'], "%Y%m%dT%H%M").strftime("%d/%m/%Y %H:%M")
        predicted_temp = predict_from_input(features)
        y_pred_xgb_orig = (predicted_temp * (params['data_temperature_max'] - params['data_temperature_min'])) + params['data_temperature_min']
        predictions.append({'timestamp': formatted_timestamp, 'temperature': y_pred_xgb_orig})
    return predictions

# Flask route to handle prediction from uploaded file
@app.route('/predict-file', methods=['POST'])
def predict_from_uploaded_file():
    try:
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return render_template('file.html', prediction_text="Không có tập tin được chọn.")
            if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
                predictions = predict_from_file(file)
                return render_template('file.html', predictions=predictions , data_weather = data_weather_to_display)
            else:
                return render_template('file.html', prediction_text="Định dạng tập tin không hợp lệ. Vui lòng tải lên tệp CSV hoặc Excel.", data_weather=data_weather_to_display)
        else:
            return render_template('file.html', prediction_text="Không có tập tin được chọn.", data_weather = data_weather_to_display)
    except Exception as e:
        return render_template('file.html', prediction_text=f"Đã xảy ra lỗi: {str(e)}", data_weather = data_weather_to_display)

if __name__ == "__main__":
    app.run(debug=True)
