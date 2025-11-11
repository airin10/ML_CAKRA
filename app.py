import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Pastikan folder upload ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Ekstensi file yang diizinkan
ALLOWED_EXTENSIONS = {'xlsx', 'xls', 'csv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_to_serializable(obj):
    """
    Fungsi untuk mengkonversi objek numpy/pandas ke tipe data Python native
    """
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif pd.isna(obj):
        return None
    else:
        return obj

def preprocess_data_advanced(df):
    """
    Fungsi preprocessing advanced sesuai dengan kode Colab
    """
    df_clean = df.copy()
    
    # =============================================================================
    # 1. 🧹 Pembersihan Data & Konversi Tipe Data Numerik
    # =============================================================================
    
    target_and_media_cols = [col for col in df_clean.columns if ' :' in col or col == 'Sales']
    other_numeric_cols = ['Contract Amount Reported']
    all_numeric_to_clean = target_and_media_cols + other_numeric_cols

    print("1a. 🔄 Mengkonversi kolom numerik yang berformat string ($ dan ,)")
    for col in all_numeric_to_clean:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip()
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # 1b. Mengisi Missing Values
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)

    object_cols = df_clean.select_dtypes(include=['object']).columns
    df_clean[object_cols] = df_clean[object_cols].fillna('Unknown')

    # =============================================================================
    # 2. 🔧 FEATURE ENGINEERING (AGGREGASI MEDIA)
    # =============================================================================

    aggregation_groups = {
        'Internet_Total': 'Internet :', 'Canvassing_Total': 'Canvassing :',
        'InPerson_Total': 'In-Person Meetings :', 'Radio_Total': 'Radio :',
        'Newspaper_Total': 'Newspaper :', 'Non-Specified Media Type_Total': 'Non-Specified Media Type :',
        'Print_Total': 'Print :', 'Television_Total': 'Television :',
        'Digital_Total': 'Digital :', 'Billboards_Total': 'Billboards :', 'Video_Total': 'Video :'
    }

    for feature_name, prefix in aggregation_groups.items():
        cols_to_sum = [col for col in df_clean.columns if isinstance(col, str) and prefix in col]
        if cols_to_sum:
            df_clean[feature_name] = df_clean[cols_to_sum].sum(axis=1)
            df_clean = df_clean.drop(columns=cols_to_sum, errors='ignore')

    df_aggregated = df_clean.copy()

    # =============================================================================
    # 3. 🎯 PEMILIHAN FITUR (REVISI: HANYA AGGREGAT MEDIA)
    # =============================================================================

    media_aggregates = list(aggregation_groups.keys())
    media_aggregates = [agg for agg in media_aggregates if agg in df_aggregated.columns]

    # --- REVISI: HANYA MENGGUNAKAN SALES DAN MEDIA AGGREGATES ---
    features_to_keep = ['Sales'] + media_aggregates

    df_model = df_aggregated[[f for f in features_to_keep if f in df_aggregated.columns]].copy()

    # =============================================================================
    # 4. 📊 PEMBAGIAN DATA & OUTLIER HANDLING (STABILISASI TARGET Y TETAP)
    # =============================================================================

    X = df_model.drop('Sales', axis=1)
    y = df_model['Sales']

    # 4a. Pembagian Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=42
    )

    # --- REVISI BARU: LOG TRANSFORM DAN WINSORIZATION PADA TARGET Y ---
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    # 4b-i. Winsorization Khusus untuk y_train_log
    y_train_proc = y_train_log.copy()
    y_lower_bound = y_train_proc.quantile(0.01)
    y_upper_bound = y_train_proc.quantile(0.99)

    y_train_log_winsorized = np.where(y_train_proc < y_lower_bound, y_lower_bound, y_train_proc)
    y_train_log_winsorized = np.where(y_train_log_winsorized > y_upper_bound, y_upper_bound, y_train_log_winsorized)

    y_train_log = pd.Series(y_train_log_winsorized, index=y_train.index)

    # 4c. Outlier Handling (Winsorization) - Hanya pada X (Fitur)
    def winsorize_data(X_df):
        X_proc = X_df.copy()
        for col in X_proc.columns:
            if X_proc[col].dtype in ['float64', 'int64']:
                lower_bound = X_proc[col].quantile(0.05)
                upper_bound = X_proc[col].quantile(0.95)
                X_proc[col] = np.where(X_proc[col] < lower_bound, lower_bound, X_proc[col])
                X_proc[col] = np.where(X_proc[col] > upper_bound, upper_bound, X_proc[col])
        return X_proc

    X_train = winsorize_data(X_train)
    X_test = winsorize_data(X_test)

    # 4d. 🧹 FINAL CHECK: MEMASTIKAN X_train/X_test HANYA BERISI NUMERIK
    def ensure_numeric_data(X_df):
        X_proc = X_df.copy()
        for col in X_proc.columns:
            X_proc[col] = pd.to_numeric(X_proc[col], errors='coerce')
            X_proc[col] = X_proc[col].fillna(0)
        return X_proc

    X_train = ensure_numeric_data(X_train)
    X_test = ensure_numeric_data(X_test)

    # =============================================================================
    # 5. ⚖️ NORMALISASI DATA (SCALING) - Hanya untuk model linear
    # =============================================================================

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    # Juga return data tanpa scaling untuk Random Forest
    X_train_df = pd.DataFrame(X_train, columns=X_train.columns, index=X_train.index)
    X_test_df = pd.DataFrame(X_test, columns=X_test.columns, index=X_test.index)

    # Informasi preprocessing untuk ditampilkan
    preprocessing_info = {
        'original_shape': df.shape,
        'cleaned_shape': df_clean.shape,
        'aggregated_shape': df_aggregated.shape,
        'final_shape': df_model.shape,
        'features_used': list(X_train.columns),
        'media_aggregates': media_aggregates,
        'y_train_stats': {
            'original_min': float(y_train.min()),
            'original_max': float(y_train.max()),
            'log_min': float(y_train_log.min()),
            'log_max': float(y_train_log.max())
        }
    }

    return (X_train_scaled_df, X_test_scaled_df, X_train_df, X_test_df, 
            y_train_log, y_test, y_test_log, scaler, preprocessing_info)

def create_binned_classification_report(y_test, y_pred):
    """
    Fungsi untuk membuat classification report via binning
    """
    try:
        # 1. Menentukan Batas Kategori (Binning)
        sales_quartiles = np.quantile(y_test, [0.25, 0.50, 0.75])
        bins = [y_test.min() - 1, sales_quartiles[0], sales_quartiles[2], y_test.max() + 1]
        labels = ['Low Sales', 'Medium Sales', 'High Sales']

        # 2. Mengubah Nilai Kontinu menjadi Kategori (Binning)
        y_test_binned = pd.Series(pd.cut(y_test, bins=bins, labels=labels, include_lowest=True, right=False)).astype(str)
        y_pred_binned = pd.Series(pd.cut(y_pred, bins=bins, labels=labels, include_lowest=True, right=False)).astype(str)

        # 3. Penanganan Missing Values/Out-of-Bound
        y_test_binned = y_test_binned.replace('nan', np.nan)
        y_pred_binned = y_pred_binned.replace('nan', np.nan)

        # Ambil Mode (kategori yang paling sering) dari nilai prediksi yang valid
        fill_value = y_pred_binned.mode()[0] if not y_pred_binned.mode().empty else 'Medium Sales'

        # Isi nilai NaN dengan Mode
        y_pred_binned = y_pred_binned.fillna(fill_value)
        y_test_binned = y_test_binned.fillna(fill_value)

        # Konversi kembali ke tipe 'category'
        y_pred_binned = y_pred_binned.astype('category')
        y_test_binned = y_test_binned.astype('category')

        # 4. Menghitung Confusion Matrix
        cm = confusion_matrix(y_test_binned, y_pred_binned, labels=labels)
        cm_df = pd.DataFrame(cm, 
                           index=[f"Aktual {l}" for l in labels], 
                           columns=[f"Prediksi {l}" for l in labels])

        # 5. Menghitung Classification Report
        report = classification_report(y_test_binned, y_pred_binned, 
                                     target_names=labels, zero_division=0, output_dict=True)

        # 6. Visualisasi Confusion Matrix (tanpa seaborn)
        plt.figure(figsize=(8, 6))
        
        # Buat heatmap manual tanpa seaborn
        plt.imshow(cm, cmap='Blues', interpolation='nearest')
        plt.colorbar()
        
        # Tambahkan annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                plt.text(j, i, str(cm[i, j]), 
                        ha='center', va='center', 
                        fontsize=12, fontweight='bold',
                        color='white' if cm[i, j] > cm.max()/2 else 'black')
        
        plt.xticks(range(len(labels)), [f'Pred {l}' for l in labels], rotation=45)
        plt.yticks(range(len(labels)), [f'Aktual {l}' for l in labels])
        plt.title('Confusion Matrix (Binned Sales)')
        plt.tight_layout()
        
        # Konversi plot ke base64
        img_cm = io.BytesIO()
        plt.savefig(img_cm, format='png', bbox_inches='tight', dpi=100)
        img_cm.seek(0)
        cm_plot_url = base64.b64encode(img_cm.getvalue()).decode()
        plt.close()

        return {
            'success': True,
            'bins_info': {
                'low_sales_max': float(sales_quartiles[0]),
                'medium_sales_min': float(sales_quartiles[0]),
                'medium_sales_max': float(sales_quartiles[2]),
                'high_sales_min': float(sales_quartiles[2])
            },
            'confusion_matrix': cm_df.to_dict(),
            'classification_report': report,
            'cm_plot': cm_plot_url,
            'labels': labels
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def create_feature_importance_plot(model, feature_names, model_type):
    """
    Fungsi untuk membuat plot feature importance
    """
    plt.figure(figsize=(10, 6))
    
    if model_type == 'random_forest' and hasattr(model, 'feature_importances_'):
        # Feature importance untuk Random Forest
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.title('Random Forest - Feature Importance')
        
    elif hasattr(model, 'coef_'):
        # Feature importance untuk model linear
        importances = np.abs(model.coef_)
        indices = np.argsort(importances)[::-1]
        
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.title('Linear Model - Feature Importance (Absolute Coefficients)')
    
    plt.tight_layout()
    
    # Konversi plot ke base64
    img_fi = io.BytesIO()
    plt.savefig(img_fi, format='png', bbox_inches='tight', dpi=100)
    img_fi.seek(0)
    fi_plot_url = base64.b64encode(img_fi.getvalue()).decode()
    plt.close()
    
    return fi_plot_url

# Fungsi untuk memproses dataset dan melatih model
def train_and_evaluate_model(file_path, model_type, file_extension):
    try:
        # Baca dataset berdasarkan ekstensi file
        if file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(file_path)
        else:  # CSV
            df = pd.read_csv(file_path)
        
        # Preprocessing data advanced
        (X_train_scaled, X_test_scaled, X_train, X_test, 
         y_train_log, y_test, y_test_log, scaler, preprocessing_info) = preprocess_data_advanced(df)
        
        # Pilih dan latih model
        if model_type == 'linear':
            model = LinearRegression()
            model.fit(X_train_scaled, y_train_log)
            # Prediksi (kembalikan ke skala asli dengan exponential)
            y_pred_log = model.predict(X_test_scaled)
            y_pred = np.expm1(y_pred_log)
            
        elif model_type == 'ridge':
            model = Ridge(alpha=1.0)
            model.fit(X_train_scaled, y_train_log)
            y_pred_log = model.predict(X_test_scaled)
            y_pred = np.expm1(y_pred_log)
            
        elif model_type == 'lasso':
            model = Lasso(alpha=1.0)
            model.fit(X_train_scaled, y_train_log)
            y_pred_log = model.predict(X_test_scaled)
            y_pred = np.expm1(y_pred_log)
            
        elif model_type == 'random_forest':
            # Random Forest tidak membutuhkan scaling
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            model.fit(X_train, y_train_log)
            y_pred_log = model.predict(X_test)
            y_pred = np.expm1(y_pred_log)
            
        else:
            model = LinearRegression()
            model.fit(X_train_scaled, y_train_log)
            y_pred_log = model.predict(X_test_scaled)
            y_pred = np.expm1(y_pred_log)
        
        # Evaluasi model regression
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Buat plot prediksi vs aktual untuk regression
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Prediksi vs Aktual (skala asli)
        plt.subplot(2, 3, 1)
        plt.scatter(y_test, y_pred, alpha=0.7, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Nilai Aktual (Sales)')
        plt.ylabel('Nilai Prediksi (Sales)')
        plt.title('Prediksi vs Aktual (Skala Asli)')
        
        # Plot 2: Residuals
        plt.subplot(2, 3, 2)
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.7, color='green')
        plt.axhline(y=0, color='red', linestyle='--')
        plt.xlabel('Prediksi')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        
        # Plot 3: Distribusi Error
        plt.subplot(2, 3, 3)
        plt.hist(residuals, bins=20, alpha=0.7, color='orange', edgecolor='black')
        plt.xlabel('Error')
        plt.ylabel('Frekuensi')
        plt.title('Distribusi Error')
        
        # Plot 4: Feature Importance
        plt.subplot(2, 3, 4)
        feature_names = X_train.columns.tolist()
        # Plot akan dibuat dalam fungsi terpisah, kita hanya perlu menampilkan gambar
        
        # Plot 5: Perbandingan Log vs Actual
        plt.subplot(2, 3, 5)
        plt.scatter(y_test_log, y_pred_log, alpha=0.7, color='red')
        plt.plot([y_test_log.min(), y_test_log.max()], [y_test_log.min(), y_test_log.max()], 'r--', lw=2)
        plt.xlabel('Nilai Aktual (Log)')
        plt.ylabel('Nilai Prediksi (Log)')
        plt.title('Prediksi vs Aktual (Skala Log)')
        
        plt.tight_layout()
        
        # Konversi plot regression ke base64
        img_reg = io.BytesIO()
        plt.savefig(img_reg, format='png', bbox_inches='tight', dpi=100)
        img_reg.seek(0)
        regression_plot_url = base64.b64encode(img_reg.getvalue()).decode()
        plt.close()
        
        # Buat feature importance plot terpisah
        feature_names = X_train.columns.tolist()
        fi_plot_url = create_feature_importance_plot(model, feature_names, model_type)
        
        # Classification Report via Binning
        binning_result = create_binned_classification_report(y_test, y_pred)
        
        # Informasi model spesifik
        model_info = {}
        if model_type == 'random_forest' and hasattr(model, 'feature_importances_'):
            # Konversi feature importances ke float
            feature_importances = {feature_names[i]: float(importance) 
                                 for i, importance in enumerate(model.feature_importances_)}
            model_info = {
                'n_estimators': int(model.n_estimators),
                'max_depth': int(model.max_depth) if model.max_depth is not None else None,
                'feature_importances': feature_importances
            }
        
        result = {
            'success': True,
            'mse': float(mse),
            'r2': float(r2),
            'regression_plot': regression_plot_url,
            'feature_importance_plot': fi_plot_url,
            'features': int(X_train.shape[1]),
            'samples': int(X_train.shape[0]),
            'file_type': file_extension.upper(),
            'preprocessing_info': preprocessing_info,
            'model_info': model_info
        }
        
        # Tambahkan hasil binning jika berhasil
        if binning_result['success']:
            result.update({
                'binning_success': True,
                'binning_info': binning_result['bins_info'],
                'confusion_matrix': binning_result['confusion_matrix'],
                'classification_report': binning_result['classification_report'],
                'cm_plot': binning_result['cm_plot'],
                'binning_labels': binning_result['labels']
            })
        else:
            result.update({
                'binning_success': False,
                'binning_error': binning_result['error']
            })
        
        # Konversi semua nilai ke serializable
        result = convert_to_serializable(result)
        
        return result
        
    except Exception as e:
        import traceback
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/evaluate', methods=['POST'])
def evaluate_model():
    # Periksa apakah file ada dalam request
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'Tidak ada file yang diupload'})
    
    file = request.files['file']
    model_type = request.form.get('model_type', 'linear')
    
    # Periksa apakah file dipilih
    if file.filename == '':
        return jsonify({'success': False, 'error': 'Tidak ada file yang dipilih'})
    
    # Periksa ekstensi file
    if not allowed_file(file.filename):
        return jsonify({
            'success': False, 
            'error': 'Format file tidak didukung. Harap upload file Excel (.xlsx, .xls) atau CSV (.csv)'
        })
    
    try:
        # Simpan file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Dapatkan ekstensi file
        file_extension = filename.rsplit('.', 1)[1].lower()
        
        # Proses file dan evaluasi model
        result = train_and_evaluate_model(file_path, model_type, file_extension)
        
        # Hapus file setelah diproses
        if os.path.exists(file_path):
            os.remove(file_path)
        
        # Konversi result ke serializable sebelum mengembalikan
        result = convert_to_serializable(result)
        return jsonify(result)
        
    except Exception as e:
        import traceback
        error_result = {
            'success': False, 
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        # Konversi error result ke serializable
        error_result = convert_to_serializable(error_result)
        return jsonify(error_result)

if __name__ == '__main__':
    app.run(debug=True)