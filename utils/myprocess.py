import pandas as pd
import torch
import joblib
from utils.Custom_modelzoo import *
from utils.predict_utils import predict_future
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, '..', 'data', 'csv', '50area_dummy_processed.csv')

df = pd.read_csv(csv_path)

def getEvse(stationname):
    filtered = df[df['station_location']==stationname]['evse_name'].unique().tolist()
    return filtered

def getStation():
    filtered = df['station_location'].unique().tolist()
    return filtered

def doInference(data):
    input_dim = 294
    hidden_dim = 128
    num_layers = 2
    num_classes = 6
    pred_len = 24
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = LSTMModel_3(input_dim, hidden_dim, num_layers, num_classes, pred_len).to(device)
    model_path = os.path.join(base_dir, '..', 'model', 'lstm_250709_152924_model_accuracy=0.4577.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))  # 실제 경로
    
    ct_path = os.path.join(base_dir, '..', 'code', 'ct_cached.joblib')
    ct = joblib.load(ct_path)
    
    station = data['충전소']
    evse = data['충전기']
    cls_preds, reg_preds = predict_future(station, evse, df, ct, model)
    return cls_preds,reg_preds
