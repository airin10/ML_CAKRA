import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

# Page configuration
st.set_page_config(
    page_title="ML Model Evaluator",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .step-card {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #28a745;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def preprocess_data_advanced(df):
    """
    Fungsi preprocessing advanced sesuai dengan kode Colab
    """
    df_clean = df.copy()
    
    # 1. Pembersihan Data & Konversi Tipe Data Numerik
    target_and_media_cols = [col for col in df_clean.columns if ' :' in col or col == 'Sales']
    other_numeric_cols = ['Contract Amount Reported']
    all_numeric_to_clean = target_and_media_cols + other_numeric_cols

    st.info("🔄 Mengkonversi kolom numerik yang berformat string ($ dan ,)")
    for col in all_numeric_to_clean:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False).str.strip()
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

    # Mengisi Missing Values
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    df_clean[numeric_cols] = df_clean[numeric_cols].fillna(0)

    object_cols = df_clean.select_dtypes(include=['object']).columns
    df_clean[object_cols] = df_clean[object_cols].fillna('Unknown')

    # 2. Feature Engineering (Aggregasi Media)
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

    # 3. Pemilihan Fitur (Hanya Aggregat Media)
    media_aggregates = list(aggregation_groups.keys())
    media_aggregates = [agg for agg in media_aggregates if agg in df_aggregated.columns]

    features_to_keep = ['Sales'] + media_aggregates
    df_model = df_aggregated[[f for f in features_to_keep if f in df_aggregated.columns]].copy()

    # 4. Pembagian Data & Outlier Handling
    X = df_model.drop('Sales', axis=1)
    y = df_model['Sales']

    # Pembagian Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Log Transform dan Winsorization pada Target Y
    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    # Winsorization Khusus untuk y_train_log
    y_train_proc = y_train_log.copy()
    y_lower_bound = y_train_proc.quantile(0.01)
    y_upper_bound = y_train_proc.quantile(0.99)

    y_train_log_winsorized = np.where(y_train_proc < y_lower_bound, y_lower_bound, y_train_proc)
    y_train_log_winsorized = np.where(y_train_log_winsorized > y_upper_bound, y_upper_bound, y_train_log_winsorized)
    y_train_log = pd.Series(y_train_log_winsorized, index=y_train.index)

    # Outlier Handling (Winsorization) - Hanya pada X (Fitur)
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

    # Final Check: Memastikan X_train/X_test hanya berisi numerik
    def ensure_numeric_data(X_df):
        X_proc = X_df.copy()
        for col in X_proc.columns:
            X_proc[col] = pd.to_numeric(X_proc[col], errors='coerce')
            X_proc[col] = X_proc[col].fillna(0)
        return X_proc

    X_train = ensure_numeric_data(X_train)
    X_test = ensure_numeric_data(X_test)

    # 5. Normalisasi Data (Scaling) - Hanya untuk model linear
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

        return {
            'success': True,
            'bins_info': {
                'low_sales_max': float(sales_quartiles[0]),
                'medium_sales_min': float(sales_quartiles[0]),
                'medium_sales_max': float(sales_quartiles[2]),
                'high_sales_min': float(sales_quartiles[2])
            },
            'confusion_matrix': cm_df,
            'classification_report': report,
            'labels': labels
        }
        
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def main():
    # Header
    st.markdown('<h1 class="main-header">🧠 ML Model Evaluator</h1>', unsafe_allow_html=True)
    st.markdown("### Upload dataset dan evaluasi model machine learning dengan berbagai algoritma")
    
    # Sidebar untuk upload dan konfigurasi
    with st.sidebar:
        st.header("ML CAKRA")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Dataset", 
            type=['xlsx', 'xls', 'csv'],
            help="Upload file Excel atau CSV. Kolom terakhir harus target variable."
        )
        
        # Model selection
        model_type = st.selectbox(
            "Pilih Model",
            ["Linear Regression", "Ridge Regression", "Lasso Regression", "Random Forest"],
            index=0
        )
        
        # Map model names to values
        model_map = {
            "Linear Regression": "linear",
            "Ridge Regression": "ridge", 
            "Lasso Regression": "lasso",
            "Random Forest": "random_forest"
        }
        
        model_value = model_map[model_type]
        
        # Analyze button
        analyze_btn = st.button("🚀 Launch Analysis", type="primary", use_container_width=True)
    
    # Main content area
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Show dataset info
            st.success(f"✅ Dataset loaded successfully: {df.shape[0]} rows × {df.shape[1]} columns")
            
            with st.expander("📊 Dataset Preview"):
                st.dataframe(df.head(), use_container_width=True)
                st.write(f"**Shape:** {df.shape}")
                st.write("**Columns:**", list(df.columns))
            
            if analyze_btn:
                with st.spinner("🔄 Training model and generating insights..."):
                    # Preprocessing
                    (X_train_scaled, X_test_scaled, X_train, X_test, 
                     y_train_log, y_test, y_test_log, scaler, preprocessing_info) = preprocess_data_advanced(df)
                    
                    # Train model based on selection
                    if model_value == 'linear':
                        model = LinearRegression()
                        model.fit(X_train_scaled, y_train_log)
                        y_pred_log = model.predict(X_test_scaled)
                    elif model_value == 'ridge':
                        model = Ridge(alpha=1.0)
                        model.fit(X_train_scaled, y_train_log)
                        y_pred_log = model.predict(X_test_scaled)
                    elif model_value == 'lasso':
                        model = Lasso(alpha=1.0)
                        model.fit(X_train_scaled, y_train_log)
                        y_pred_log = model.predict(X_test_scaled)
                    else:  # random_forest
                        model = RandomForestRegressor(
                            n_estimators=100,
                            max_depth=10,
                            min_samples_split=5,
                            min_samples_leaf=2,
                            random_state=42
                        )
                        model.fit(X_train, y_train_log)
                        y_pred_log = model.predict(X_test)
                    
                    # Convert predictions back to original scale
                    y_pred = np.expm1(y_pred_log)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Display Results
                    st.success("🎉 Analysis Complete!")
                    
                    # Metrics in columns
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Model", model_type)
                    with col2:
                        st.metric("Mean Squared Error", f"{mse:.4f}")
                    with col3:
                        st.metric("R-squared (R²)", f"{r2:.4f}")
                    with col4:
                        st.metric("Features Used", len(preprocessing_info['features_used']))
                    
                    # Preprocessing Steps
                    st.subheader("🔧 Preprocessing Steps")
                    
                    steps_col1, steps_col2 = st.columns(2)
                    
                    with steps_col1:
                        st.markdown("""
                        <div class="step-card">
                            <h4>🧹 Data Cleaning</h4>
                            <p>• Removed $ symbols and commas</p>
                            <p>• Handled missing values</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("""
                        <div class="step-card">
                            <h4>🎯 Feature Selection</h4>
                            <p>Selected media aggregate features</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with steps_col2:
                        st.markdown("""
                        <div class="step-card">
                            <h4>🔧 Feature Engineering</h4>
                            <p>Aggregated media channels</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("""
                        <div class="step-card">
                            <h4>📊 Data Transformation</h4>
                            <p>• Log transform</p>
                            <p>• Winsorization</p>
                            <p>• Standard scaling</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Visualizations
                    st.subheader("📈 Visualizations")
                    
                    # Create plots
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
                    
                    # Plot 1: Actual vs Predicted
                    ax1.scatter(y_test, y_pred, alpha=0.7, color='blue')
                    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                    ax1.set_xlabel('Actual Values')
                    ax1.set_ylabel('Predicted Values')
                    ax1.set_title('Actual vs Predicted')
                    
                    # Plot 2: Residuals
                    residuals = y_test - y_pred
                    ax2.scatter(y_pred, residuals, alpha=0.7, color='green')
                    ax2.axhline(y=0, color='red', linestyle='--')
                    ax2.set_xlabel('Predicted Values')
                    ax2.set_ylabel('Residuals')
                    ax2.set_title('Residual Plot')
                    
                    # Plot 3: Feature Importance
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = model.feature_importances_
                        features = X_train.columns
                        indices = np.argsort(feature_importance)[::-1]
                        ax3.bar(range(len(feature_importance)), feature_importance[indices])
                        ax3.set_xticks(range(len(feature_importance)))
                        ax3.set_xticklabels([features[i] for i in indices], rotation=45)
                        ax3.set_title('Feature Importance')
                    else:
                        # For linear models, show coefficients
                        if hasattr(model, 'coef_'):
                            coefficients = np.abs(model.coef_)
                            features = X_train.columns
                            indices = np.argsort(coefficients)[::-1]
                            ax3.bar(range(len(coefficients)), coefficients[indices])
                            ax3.set_xticks(range(len(coefficients)))
                            ax3.set_xticklabels([features[i] for i in indices], rotation=45)
                            ax3.set_title('Feature Importance (Absolute Coefficients)')
                    
                    # Plot 4: Error Distribution
                    ax4.hist(residuals, bins=20, alpha=0.7, color='orange', edgecolor='black')
                    ax4.set_xlabel('Error')
                    ax4.set_ylabel('Frequency')
                    ax4.set_title('Error Distribution')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Binning Classification Report
                    st.subheader("📊 Classification Report (Binned Sales)")
                    
                    binning_result = create_binned_classification_report(y_test, y_pred)
                    
                    if binning_result['success']:
                        binning_info = binning_result['bins_info']
                        
                        st.info(f"""
                        **Sales Categories:**
                        - **Low Sales:** < ${binning_info['low_sales_max']:,.2f}
                        - **Medium Sales:** ${binning_info['medium_sales_min']:,.2f} - ${binning_info['medium_sales_max']:,.2f}
                        - **High Sales:** > ${binning_info['high_sales_min']:,.2f}
                        """)
                        
                        # Confusion Matrix
                        cm_df = binning_result['confusion_matrix']
                        fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                        ax_cm.set_title('Confusion Matrix')
                        st.pyplot(fig_cm)
                        
                        # Classification Report
                        report = binning_result['classification_report']
                        st.subheader("Classification Metrics")
                        report_df = pd.DataFrame(report).transpose()
                        st.dataframe(report_df, use_container_width=True)
                    
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.info("Please check your dataset format and try again.")
    
    else:
        # Welcome message when no file uploaded
        st.info("👆 Please upload a dataset to get started")
        
        # Sample workflow
        with st.expander("ℹ️ How to use this app"):
            st.markdown("""
            1. **Upload** your Excel or CSV file
            2. **Select** a machine learning model
            3. **Click** "Launch Analysis"
            4. **View** results and visualizations
            
            **Requirements:**
            - Last column should be the target variable (Sales)
            - Supported formats: .xlsx, .xls, .csv
            - Data will be automatically preprocessed
            """)

if __name__ == "__main__":
    main()