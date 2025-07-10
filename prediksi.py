import pandas as pd
import numpy as np
import streamlit as st
import os
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from collections import Counter

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Diabetes", layout="wide")
st.title("Prediksi Diabetes Para Pasien")

# Masukkan Dataset
def load_data():
    df = pd.read_csv('Dataset_Repeel.txt', delimiter=';')
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['Outcome'])
    df['Outcome'] = df['Outcome'].astype(int)
    return df

try:
    df = load_data()
    st.success("✅ Dataset berhasil dimuat")
except Exception as e:
    st.error(f"❌ Error saat memuat dataset: {e}")
    st.stop()

# Mendefinisikan fitur dan target
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

X = df[features].fillna(df[features].mean())
y = df['Outcome']

# Train model
st.subheader("Training Model Machine Learning")

if st.button("Train Model dan Log ke MLflow"):
    # Hitung distribusi kelas untuk menentukan parameter SMOTE
    count = Counter(y)
    min_class_count = min(count.values())

    try:
        k_neighbors = min(5, min_class_count - 1)
        smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        st.info("✅ SMOTE berhasil diterapkan untuk menyeimbangkan data")
        use_class_weight = False
    except Exception as e:
        st.warning(f"⚠️ Gagal menjalankan SMOTE: {e}")
        st.info("Menggunakan class_weight='balanced' sebagai alternatif.")
        X_resampled, y_resampled = X, y
        use_class_weight = True

    # Validasi data target
    if pd.isnull(y_resampled).any():
        st.error("❌ Target y mengandung nilai NaN. Training dibatalkan.")
        st.stop()

    # Membagi data training dan data testing
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    # Pilih classifier berdasarkan apakah menggunakan class_weight
    if use_class_weight:
        classifier = RandomForestClassifier(random_state=42, class_weight="balanced")
    else:
        classifier = RandomForestClassifier(random_state=42)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])

    # Membuat parameter grid untuk tuning
    param_grid = {
        'classifier__n_estimators': [100],
        'classifier__max_depth': [None],
        'classifier__min_samples_split': [2]
    }

    # Setup GridSearchCV untuk mencari parameter terbaik
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1
    )

    mlflow.set_experiment("Diabetes_Prediction")

    # Mulai tracking dengan MLflow
    with mlflow.start_run():
        grid_search.fit(X_train, y_train)
        model = grid_search.best_estimator_

        # Evaluasi model
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Menyimpan log parameter dan metrik ke MLflow
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        st.success(f"🎯 Akurasi Model: {acc * 100:.2f}%")
        st.info("✅ Model berhasil dicatat ke MLflow")

        # Simpan model
        joblib.dump(model, "models/diabetes_model.pkl")
        st.info("💾 Model disimpan ke file models/diabetes_model.pkl")

# Cek folder apakah model sudah ada
if os.path.exists('models/diabetes_model.pkl'):
    model = joblib.load('models/diabetes_model.pkl')
    st.success("✅ Model berhasil dimuat")
else:
    st.warning("⚠️ Model belum dilatih. Jalankan training terlebih dahulu.")

# Form pasien baru
if 'model' in locals():
    st.subheader("📝 Form Pasien Baru")

    with st.form("input_pasien"):
        cols = st.columns(3)
        input_data = {}

        default_values = {
            'Pregnancies': 0.0,
            'Glucose': 0.0,
            'BloodPressure': 0.0,
            'SkinThickness': 0.0,
            'Insulin': 0.0,
            'BMI': 0.0,
            'DiabetesPedigreeFunction': 0.0,
            'Age': 0.0
        }

        for idx, feature in enumerate(features):
            with cols[idx % 3]:
                input_str = st.text_input(
                    f"{feature}",
                    value=str(default_values[feature]).replace('.', ','),
                    key=f"input_{feature}"
                )
                try:
                    input_data[feature] = float(input_str.replace(',', '.'))
                except ValueError:
                    input_data[feature] = default_values[feature]
                    st.warning(f"Nilai {feature} tidak valid, menggunakan default {default_values[feature]}")

        pasien_id = st.text_input("Masukkan ID Pasien", value="")
        submitted = st.form_submit_button("Prediksi dan Simpan ke Dataset")

    if submitted:
        if pasien_id.strip() == "":
            st.warning("⚠️ Harap masukkan ID pasien terlebih dahulu.")
        else:
            try:
                df_new = pd.DataFrame([input_data])

                if hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
                    scaled = model.named_steps['scaler'].transform(df_new)
                else:
                    scaled = df_new

                # Lakukan prediksi
                hasil = model.predict(scaled)
                prob = model.predict_proba(scaled)

                # Tampilkan hasil prediksi
                st.markdown("---")
                st.subheader("🩺 Hasil Prediksi Diabetes")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("ID Pasien", pasien_id)

                with col2:
                    st.metric("Tidak Akan Memiliki Diabetes",
                              f"{prob[0][0] * 100:.1f}%")

                with col3:
                    st.metric("Akan Memiliki Diabetes",
                              f"{prob[0][1] * 100:.1f}%")

                # Siapkan data untuk disimpan
                input_data['id'] = pasien_id
                input_data['Outcome'] = int(hasil[0])
                input_data['Prediksi_Model'] = "Akan Memiliki Diabetes" if hasil[
                                                                        0] == 1 else "Tidak Akan Memiliki Diabetes"
                input_data['Probabilitas_Tidak_Diabetes'] = prob[0][0]
                input_data['Probabilitas_Diabetes'] = prob[0][1]

                # Gabungkan dengan data existing
                df_append = pd.DataFrame([input_data])

                try:
                    df_existing = pd.read_csv('diabetes.csv', delimiter=';')
                    df_combined = pd.concat([df_existing, df_append], ignore_index=True)
                    df_combined.to_csv('diabetes.csv', index=False, sep=';')
                    st.success(f"✅ Data pasien dengan ID `{pasien_id}` berhasil disimpan ke dataset!")
                except Exception as e:
                    st.error(f"❌ Gagal menyimpan data: {e}")

            except Exception as e:
                st.error(f"❌ Terjadi error saat memproses prediksi: {e}")

# Mencari Pasien
st.subheader("🔍 Cari Pasien")
search_id = st.text_input("Masukkan ID pasien untuk dicek")

if st.button("Cari Pasien"):
    if search_id.strip() == "":
        st.warning("⚠️ Harap masukkan ID pasien.")
    else:
        try:
            # Cari data pasien
            df_check = pd.read_csv('diabetes.csv', delimiter=';')
            result = df_check[df_check['id'].astype(str) == search_id.strip()]

            if not result.empty:
                # Tampilkan format
                result_display = result.copy()
                for col in result_display.columns:
                    if result_display[col].dtype == float:
                        result_display[col] = result_display[col].round(2)

                # Tampilkan hasil
                st.subheader(f"Hasil Pencarian ID: {search_id}")
                st.dataframe(result_display)

                # Tampilkan prediksi
                prediksi = result_display.iloc[0]['Prediksi_Model']
                prob_diabetes = result_display.iloc[0]['Probabilitas_Diabetes'] * 100

                if prediksi == "Akan Memiliki Diabetes":
                    st.error(f"⚠️ Pasien ini diprediksi {prediksi} (Probabilitas: {prob_diabetes:.1f}%)")
                else:
                    st.success(f"✅ Pasien ini diprediksi {prediksi} (Probabilitas: {prob_diabetes:.1f}%)")
            else:
                st.error(f"❌ Data dengan ID `{search_id}` tidak ditemukan.")
        except Exception as e:
            st.error(f"❌ Error saat mencari data: {e}")

# Analisis dataset
if 'model' in locals():
    df = load_data()
    X = df[features].fillna(df[features].mean())

    # Preprocess data untuk prediksi full dataset
    if hasattr(model, 'named_steps') and 'scaler' in model.named_steps:
        X_scaled = model.named_steps['scaler'].transform(X)
    else:
        X_scaled = X

    # Prediksi untuk seluruh dataset
    y_pred_full = model.predict(X_scaled)
    y_prob_full = model.predict_proba(X_scaled)

    # Tambahkan kolom prediksi ke dataframe
    df['Status_Sekarang'] = df['Outcome'].map({1: 'Memiliki Diabetes', 0: 'Tidak Memiliki Diabetes'})
    df['Prediksi_Model'] = np.where(y_pred_full == 1, 'Akan Memiliki Diabetes', 'Tidak Akan Memiliki Diabetes')
    df['Probabilitas_Diabetes'] = y_prob_full[:, 1]
    df['Probabilitas_Tidak_Diabetes'] = y_prob_full[:, 0]

    # Tampilkan analisis dataset
    st.subheader("Analisis Dataset Lengkap")

    tab1, tab2 = st.tabs(["Data", "Statistik & Visualisasi"])

    with tab1:
        st.dataframe(df)  # Tampilkan dataframe lengkap

    with tab2:
        st.write("#### Distribusi Status dan Visualisasi")
        col1, col2 = st.columns(2)

        with col1:
            # Tampilkan distribusi status aktual
            st.write("**Status Aktual**")
            status_counts = df['Status_Sekarang'].value_counts()
            st.dataframe(status_counts)

            # Tampilkan distribusi prediksi
            st.write("**Hasil Prediksi Model**")
            pred_counts = df['Prediksi_Model'].value_counts()
            st.dataframe(pred_counts)

            # Hitung akurasi prediksi vs aktual
            st.write("#### Akurasi Prediksi vs Aktual")
            df['Prediksi_Benar'] = (df['Status_Sekarang'] == 'Memiliki Diabetes') == (
                    df['Prediksi_Model'] == 'Akan Memiliki Diabetes')
            accuracy = df['Prediksi_Benar'].mean()
            st.metric("Akurasi Prediksi", f"{accuracy * 100:.1f}%")

        with col2:
            # Buat visualisasi pie chart berdampingan
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))  # 1 baris, 2 kolom

            # Pie chart status aktual
            status_counts.plot.pie(
                ax=axes[0],
                labels=status_counts.index,
                autopct='%1.1f%%',
                colors=['#ADD8E6', '#FA8072'],
                textprops={'fontsize': 8},
                startangle=90
            )
            axes[0].set_title('Status Diabetes Aktual', fontsize=10)
            axes[0].set_ylabel("")

            # Pie chart hasil prediksi
            pred_counts.plot.pie(
                ax=axes[1],
                labels=pred_counts.index,
                autopct='%1.1f%%',
                colors=['#ADD8E6', '#FA8072'],
                textprops={'fontsize': 8},
                startangle=90
            )
            axes[1].set_title('Prediksi Diabetes oleh Model', fontsize=10)
            axes[1].set_ylabel("")

            plt.tight_layout()
            st.pyplot(fig)

# Download dataset
st.subheader("📥 Download Data")
csv = df.to_csv(index=False).encode('utf-8')
st.download_button(
    label="Download CSV Lengkap",
    data=csv,
    file_name='data_diabetes_lengkap.csv',
    mime='text/csv',
)