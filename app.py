from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import warnings
warnings.filterwarnings('ignore')

# =========================
# DEPLOY STABILITY FIX
# =========================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = Flask(__name__)

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
model_ann = None
df_global = None
kabupaten_list = []


# =========================
# TRAIN MODEL
# =========================
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

    history = model.fit(
        X_train, y_train,
        epochs=50,
        validation_data=(X_test, y_test),
        verbose=0
    )

    loss, mae = model.evaluate(X_test, y_test, verbose=0)

    model_ann = model

    return model, history, mae


# =========================
# HOME
# =========================
@app.route('/')
def index():
    return render_template('index.html', kabupaten_list=kabupaten_list)


# =========================
# LOAD DATA
# =========================
@app.route('/load_data', methods=['POST'])
def load_data():
    global df_global, kabupaten_list

    try:
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            df = pd.read_csv(file)
        else:
            path = os.path.join(os.path.dirname(__file__), 'Kepadatan_penduduk_jabar.csv')
            if not os.path.exists(path):
                return jsonify({'status': 'error', 'message': 'CSV tidak ditemukan'})
            df = pd.read_csv(path)

        df.columns = [c.lower().strip() for c in df.columns]

        required = ['nama_kabupaten_kota', 'kepadatan_penduduk', 'tahun']
        for c in required:
            if c not in df.columns:
                return jsonify({'status': 'error', 'message': f'Kolom {c} tidak ditemukan'})

        df_global = df.copy()
        kabupaten_list = sorted(df['nama_kabupaten_kota'].unique().tolist())

        return jsonify({
            'status': 'success',
            'stats': {
                'total_rows': len(df),
                'total_kabupaten': len(kabupaten_list),
                'tahun_range': f"{int(df['tahun'].min())} - {int(df['tahun'].max())}",
                'kabupaten_list': kabupaten_list
            }
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


# =========================
# TRAIN
# =========================
@app.route('/train', methods=['POST'])
def train():
    global df_global

    if df_global is None:
        return jsonify({'status': 'error', 'message': 'Upload data dulu'})

    data = request.json
    kabupaten = data.get('kabupaten', '')

    df_filtered = df_global[df_global['nama_kabupaten_kota'] == kabupaten].copy()
    df_filtered = df_filtered.sort_values('tahun')

    if len(df_filtered) < 3:
        return jsonify({'status': 'error', 'message': 'Data kurang'})

    model, history, mae = load_and_train(df_filtered)

    actual = [
        {'tahun': int(r['tahun']), 'kepadatan': float(r['kepadatan_penduduk'])}
        for _, r in df_filtered.iterrows()
    ]

    return jsonify({
        'status': 'success',
        'mae': round(float(mae), 4),
        'actual_data': actual,
        'loss_history': [float(x) for x in history.history['loss'][::2]],
        'epochs_sample': list(range(len(history.history['loss'][::2])))
    })


# =========================
# PREDICT
# =========================
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

    X_future = np.array(tahun_list).reshape(-1, 1)
    X_future_scaled = scaler_X.transform(X_future)

    pred_scaled = model_ann.predict(X_future_scaled, verbose=0)
    pred = scaler_y.inverse_transform(pred_scaled)

    predictions = [
        {'tahun': int(t), 'prediksi': round(float(p), 2)}
        for t, p in zip(tahun_list, pred.flatten())
    ]

    X_all = df_filtered['tahun'].values.reshape(-1, 1)
    X_all_scaled = scaler_X.transform(X_all)
    pred_all = model_ann.predict(X_all_scaled, verbose=0)
    pred_all = scaler_y.inverse_transform(pred_all).flatten()

    overlay = [
        {'tahun': int(t), 'prediksi': round(float(p), 2)}
        for t, p in zip(df_filtered['tahun'].values, pred_all)
    ]

    return jsonify({
        'status': 'success',
        'predictions': predictions,
        'overlay': overlay
    })


# =========================
# MAIN (FIX DEPLOY RAILWAY)
# =========================
if __name__ == "__main__":
    try:
        path = os.path.join(os.path.dirname(__file__), 'Kepadatan_penduduk_jabar.csv')
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.columns = [c.lower().strip() for c in df.columns]
            df_global = df.copy()
            kabupaten_list = sorted(df['nama_kabupaten_kota'].unique().tolist())
            print(f"✅ Data loaded: {len(kabupaten_list)} kabupaten/kota")
    except Exception as e:
        print("⚠️ Error:", e)

    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)