import json, os

def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)

def load_data_weather():
    return read_json('D:\DATN\Test-UI-2\data\data_weather.json')