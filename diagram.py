import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import shap

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Buat direktori untuk menyimpan hasil output
os.makedirs('plots', exist_ok=True)
os.makedirs('reports', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Inisialisasi file laporan evaluasi model
report_file = 'reports/model_evaluation.txt'

def write_report(content):
    with open(report_file, 'a') as f:
        f.write(str(content) + "\n")

# Buat header laporan
with open(report_file, 'w') as f:
    f.write("EVALUASI MODEL DIABETES\n")
    f.write("=" * 50 + "\n\n")

# Membaca dataset
df = pd.read_csv('Dataset_Repeel.txt', delimiter=';')
df.columns = df.columns.str.strip()

# Pemeriksaan values yang hilang ke laporan
write_report("JUMLAH MISSING VALUES PER KOLOM")
write_report(df.isna().sum())

# Mendefinisikan fitur dan target
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X = df[features]
y = df['Outcome']

# Pembersihan data dengan menghapus baris dari values yang hilang
df_clean = df.dropna(subset=['Outcome'])
X_clean = df_clean[features]
y_clean = df_clean['Outcome']
X, y = X_clean, y_clean

# Visualisasi 1 : scatter umur dengan glukosa
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_clean, x='Age', y='Glucose', hue='Outcome', palette={0.0: 'lightblue', 1.0: 'salmon'})
plt.title('Age vs Glucose by Outcome')
plt.tight_layout()
plt.savefig('plots/scatter.png', dpi=300)
plt.close()

# Visualisasi 2 : hasil distribusi
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
outcome_counts = y.value_counts().sort_index()
fig.suptitle('Hasil Distribusi', fontsize=16)

# Hasil berupa pie chart
axes[0].pie(outcome_counts, labels=['Non-Diabetes', 'Diabetes'],
            autopct='%1.1f%%', colors=['lightblue', 'salmon'], startangle=90)

# Hasil berupa bar chart
sns.barplot(x=outcome_counts.index, y=outcome_counts.values,
            ax=axes[1], palette=['lightblue', 'salmon'])
axes[1].set_xticklabels(['Non-Diabetes', 'Diabetes'])
axes[1].set_ylabel('Jumlah')
axes[1].bar_label(axes[1].containers[0], fmt='%d')

plt.tight_layout()
plt.savefig('plots/hasil_distribusi.png', dpi=300)
plt.close()

# Visualisasi 3 : distribusi semua fitur
plt.figure(figsize=(14, 12))
for i, feature in enumerate(features, 1):
    plt.subplot(3, 3, i)
    ax = sns.histplot(data=df_clean, x=feature, kde=True, bins=15, shrink=0.85)
    plt.title(f'Distribusi {feature}')

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height)}',
                        (p.get_x() + p.get_width() / 2, height),
                        ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('plots/fitur_distribusi.png', dpi=300)
plt.close()

# Visualisasi 4 : kolerasi matrix
plt.figure(figsize=(10, 8))
corr_matrix = df_clean[features + ['Outcome']].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Korelasi Matriks')
plt.tight_layout()
plt.savefig('plots/correlation_matrix.png', dpi=300)
plt.close()

# Pemrosesan Data (SMOTE)
write_report("\nPEMROSESAN DATA (SMOTE)")
write_report("Menerapkan SMOTE untuk menangani imbalance class...")

try:
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    write_report("Setelah SMOTE:")
    write_report(pd.Series(y_res).value_counts())
except Exception as e:
    write_report(f"Error dalam SMOTE: {str(e)}")
    raise

# Pembuatan Model
# Membuat data menjadi data train dan data testing
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

# Membuat pipeline dengan random forest
pipeline = Pipeline([
    ('classifier', RandomForestClassifier(random_state=42))
])

# Parameter grid untuk tuning
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}
# Grid search untuk mencari parameter terbaik
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

# Ambil model terbaik
model = grid.best_estimator_

# Evaluasi Model
write_report("\nEVALUASI MODEL")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# Tulis hasil ke laporan
write_report(f"Akurasi: {acc:.4f}")
write_report("Confusion Matrix:")
write_report(confusion_matrix(y_test, y_pred))
write_report("Classification Report:")
write_report(classification_report(y_test, y_pred))

# Analisis fitur penting
importances = model.named_steps['classifier'].feature_importances_
feature_importance = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(12, 6))
ax = feature_importance.sort_values().plot.barh(color='teal')
plt.title('Feature Importance')

for i in ax.patches:
    plt.text(i.get_width() + 0.001, i.get_y() + i.get_height()/2,
             f'{i.get_width():.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('plots/fitur_penting.png', dpi=300)
plt.close()


# Menganalisis SHAP
try:
    explainer = shap.TreeExplainer(model.named_steps['classifier'])
    shap_values = explainer.shap_values(X_test)

    plt.figure()
    shap.summary_plot(shap_values[1], X_test, feature_names=features, show=False)
    plt.tight_layout()
    plt.savefig('plots/shap_summary.png', bbox_inches='tight', dpi=300)
    plt.close()
except Exception as e:
    print(f"Error dalam membuat SHAP plot: {str(e)}")

# Menyimpan model ke file
model_path = 'models/Model_Repeel.pkl'
joblib.dump(model, model_path)

# Pemberitahuan
write_report("\nSELESAI")
write_report(f"Akurasi akhir model: {acc:.4f}")
print(f"\n✅ Proses selesai! Laporan tersedia di: {report_file}")
