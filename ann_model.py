# ============================================
# 1. Import Library
# ============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ============================================
# 2. Membaca Data dari File CSV
# ============================================
df_raw = pd.read_csv("Kepadatan_penduduk_jabar.csv")

# Agregasi ke rata-rata kepadatan per tahun
df = df_raw.groupby('tahun')['kepadatan_penduduk'].mean().reset_index()
df.columns = ['Tahun', 'Penduduk']

print("Data yang digunakan:")
print(df)
print(f"\nTotal data: {len(df)} tahun")

# ============================================
# 3. Visualisasi Data (GRAFIK 1)
# ============================================
plt.figure(figsize=(8,5))
sns.scatterplot(x=df["Tahun"], y=df["Penduduk"], color="blue", label="Data Aktual")
plt.xlabel("Tahun")
plt.ylabel("Rata-rata Kepadatan Penduduk (jiwa/km²)")
plt.title("Pertumbuhan Kepadatan Penduduk Jawa Barat")
plt.legend()
plt.show()

# ============================================
# 4. Preprocessing Data
# ============================================
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[["Tahun", "Penduduk"]])

X = df_scaled[:, 0].reshape(-1, 1)  # Tahun
Y = df_scaled[:, 1]                  # Penduduk

# Split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(f"\nData training: {len(X_train)} tahun")
print(f"Data testing: {len(X_test)} tahun")

# ============================================
# 5. Membangun Model ANN
# ============================================
model = Sequential([
    Dense(10, activation='relu', input_shape=(1,)),
    Dense(10, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(X_train, Y_train, epochs=200, validation_data=(X_test, Y_test), verbose=0)

# ============================================
# 6. Evaluasi Model
# ============================================
loss, mae = model.evaluate(X_test, Y_test, verbose=0)
print(f"\nMean Absolute Error (MAE): {mae:.4f}")

# ============================================
# 7. Prediksi Tahun Mendatang
# ============================================
tahun_prediksi = np.array([[2026], [2027], [2028], [2029], [2030]])
tahun_prediksi_scaled = scaler.transform(np.column_stack((tahun_prediksi, np.zeros(len(tahun_prediksi)))))[:, 0].reshape(-1, 1)
prediksi_scaled = model.predict(tahun_prediksi_scaled, verbose=0)
prediksi = scaler.inverse_transform(np.column_stack((tahun_prediksi_scaled[:, 0], prediksi_scaled)))[:, 1]

print("\nHasil Prediksi:")
for tahun, penduduk in zip([2026, 2027, 2028, 2029, 2030], prediksi):
    print(f"Tahun {tahun}: {int(penduduk)} jiwa/km²")

# ============================================
# 8. Visualisasi Hasil Prediksi vs Data Aktual (GRAFIK 2) - DIPERBAIKI
# ============================================
# Prediksi untuk ALL data (bukan hanya data uji)
X_all_denorm = df["Tahun"].values
Y_all_denorm = df["Penduduk"].values

# Prediksi untuk semua tahun yang ada di dataset
X_all_scaled = scaler.transform(np.column_stack((X_all_denorm.reshape(-1,1), np.zeros(len(X_all_denorm)))))[:, 0].reshape(-1,1)
Y_all_pred_scaled = model.predict(X_all_scaled, verbose=0)
Y_all_pred_denorm = scaler.inverse_transform(np.column_stack((X_all_scaled[:, 0], Y_all_pred_scaled)))[:, 1]

# Plot SEMUA data aktual vs prediksi
plt.figure(figsize=(10,6))
plt.scatter(X_all_denorm, Y_all_denorm, color='blue', s=100, label="Data Aktual (Semua)")
plt.scatter(X_all_denorm, Y_all_pred_denorm, color='red', s=100, marker='^', label="Prediksi ANN (Semua)")
plt.xlabel("Tahun", fontsize=12)
plt.ylabel("Rata-rata Kepadatan Penduduk (jiwa/km²)", fontsize=12)
plt.title("Hasil Prediksi ANN vs Data Aktual", fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.show()