from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
model_ann = None
df_global = None
kabupaten_list = []


def load_and_train(df_filtered):
    global model_ann, scaler_X, scaler_y
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X = df_filtered[["tahun"]].values
    y = df_filtered[["kepadatan_penduduk"]].values

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_scaled, test_size=0.2, random_state=42
    )

    model = Sequential([
        Dense(10, activation='relu', input_shape=(1,)),
        Dense(10, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    history = model.fit(X_train, y_train, epochs=200,
                        validation_data=(X_test, y_test), verbose=0)

    loss, mae = model.evaluate(X_test, y_test, verbose=0)
    model_ann = model
    return model, history, mae, X_test, y_test


@app.route('/')
def index():
    return render_template('index.html', kabupaten_list=kabupaten_list)


@app.route('/load_data', methods=['POST'])
def load_data():
    global df_global, kabupaten_list

    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        df = pd.read_csv(file)
    else:
        default_path = os.path.join(os.path.dirname(__file__), 'Kepadatan_penduduk_jabar.csv')
        if os.path.exists(default_path):
            df = pd.read_csv(default_path)
        else:
            return jsonify({'status': 'error', 'message': 'File tidak ditemukan'})

    df.columns = [c.lower().strip() for c in df.columns]

    required = ['nama_kabupaten_kota', 'kepadatan_penduduk', 'tahun']
    for col in required:
        if col not in df.columns:
            return jsonify({'status': 'error', 'message': f'Kolom {col} tidak ditemukan'})

    df_global = df.copy()
    kabupaten_list = sorted(df['nama_kabupaten_kota'].unique().tolist())

    stats = {
        'total_rows': len(df),
        'total_kabupaten': len(kabupaten_list),
        'tahun_range': f"{int(df['tahun'].min())} - {int(df['tahun'].max())}",
        'kabupaten_list': kabupaten_list
    }
    return jsonify({'status': 'success', 'stats': stats})


@app.route('/train', methods=['POST'])
def train():
    global df_global

    if df_global is None:
        return jsonify({'status': 'error', 'message': 'Upload data terlebih dahulu'})

    data = request.json
    kabupaten = data.get('kabupaten', '')

    df_filtered = df_global[df_global['nama_kabupaten_kota'] == kabupaten].copy()
    df_filtered = df_filtered.sort_values('tahun')

    if len(df_filtered) < 3:
        return jsonify({'status': 'error', 'message': 'Data tidak cukup'})

    model, history, mae, X_test, y_test = load_and_train(df_filtered)

    actual_data = []
    for _, row in df_filtered.iterrows():
        actual_data.append({'tahun': int(row['tahun']), 'kepadatan': float(row['kepadatan_penduduk'])})

    loss_history = history.history['loss'][:200:2]

    return jsonify({
        'status': 'success',
        'mae': round(float(mae), 4),
        'actual_data': actual_data,
        'loss_history': [round(v, 6) for v in loss_history],
        'epochs_sample': list(range(0, 200, 2))
    })


@app.route('/predict', methods=['POST'])
def predict():
    global model_ann, scaler_X, scaler_y, df_global

    if model_ann is None:
        return jsonify({'status': 'error', 'message': 'Model belum dilatih'})

    data = request.json
    kabupaten = data.get('kabupaten', '')
    tahun_list = data.get('tahun_list', [2026, 2027, 2028, 2029, 2030])

    df_filtered = df_global[df_global['nama_kabupaten_kota'] == kabupaten].copy()
    df_filtered = df_filtered.sort_values('tahun')

    tahun_arr = np.array(tahun_list).reshape(-1, 1)
    tahun_scaled = scaler_X.transform(tahun_arr)
    pred_scaled = model_ann.predict(tahun_scaled, verbose=0)
    pred = scaler_y.inverse_transform(pred_scaled)

    results = []
    for t, p in zip(tahun_list, pred.flatten()):
        results.append({'tahun': int(t), 'prediksi': round(float(p), 2)})

    all_tahun = df_filtered['tahun'].values.reshape(-1, 1)
    all_scaled = scaler_X.transform(all_tahun)
    all_pred_scaled = model_ann.predict(all_scaled, verbose=0)
    all_pred = scaler_y.inverse_transform(all_pred_scaled).flatten()

    overlay = []
    for t, p in zip(df_filtered['tahun'].values, all_pred):
        overlay.append({'tahun': int(t), 'prediksi': round(float(p), 2)})

    return jsonify({
        'status': 'success',
        'predictions': results,
        'overlay': overlay
    })


if __name__ == '__main__':
    default_path = os.path.join(os.path.dirname(__file__), 'Kepadatan_penduduk_jabar.csv')
    if os.path.exists(default_path):
        df = pd.read_csv(default_path)
        df.columns = [c.lower().strip() for c in df.columns]
        df_global = df.copy()
        kabupaten_list = sorted(df['nama_kabupaten_kota'].unique().tolist())
        print(f"✅ Data loaded: {len(kabupaten_list)} kabupaten/kota")

    app.run(debug=True, port=5000)