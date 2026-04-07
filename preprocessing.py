import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from pathlib import Path
import joblib

def preprocess_data(raw_path, processed_dir):
    raw_file_path = r"C:\Users\hp\Desktop\Deeplearning\Soil_Erosion_Project_DS\data\raw\Merged_Woredas_All.xlsx"
    df = pd.read_excel(raw_path)
    print(f"Initial shape: {df.shape}")
    
    # 1. MISSING VALUES: Impute median for numerics, mode for cats
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    imputer_num = SimpleImputer(strategy='median')
    df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)
    
    print(f"After imputation: {df.isnull().sum().sum()} missing")
    
    # 2. OUTLIERS: IQR method + domain caps
    # Clip extreme values (e.g., NDVI -1 to 1, Slope 0-60°, curvatures)
    df['NDVI_Value'] = np.clip(df['NDVI_Value'], -0.1, 1.0)
    df['Slope (Degree)'] = np.clip(df['Slope (Degree)'], 0, 60)
    df['Rainfall (mm)'] = np.clip(df['Rainfall (mm)'], 300, 2000)  # Ethiopia range
    
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Winsorize (cap) outliers instead of drop for geospatial data
    for col in ['Drainage Density (m)', 'Elevation (m)', 'SPI', 'TPI', 'TRI', 'TWI']:
        df[col] = np.clip(df[col], lower_bound[col], upper_bound[col])
    
    # Log transform remaining skew (curvatures, TRI)
    skew_cols = ['Plan Curvature', 'Profile Curvature', 'TRI']
    for col in skew_cols:
        if col in df.columns:
            df[col] = np.log1p(np.abs(df[col]))
    
    print(f"Outliers winsorized: Slope max={df['Slope (Degree)'].max():.1f}, NDVI range={df['NDVI_Value'].min():.2f}-{df['NDVI_Value'].max():.2f}")
    
    # 3. DUPLICATES: Remove exact dupes (rare in geo data)
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Removed {initial_rows - len(df)} duplicates")
    
    # 4. ENCODE CATEGORICALS
    encoders = {}
    categoricals = ['Soil Type', 'Geology_Formation', 'Land_Use', 'Woreda']
    for col in categoricals:
        le = LabelEncoder()
        df[f'{col}_enc'] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    
    # 5. ENGINEER TARGET: RUSLE-inspired erosion_risk
    df['erosion_risk'] = (
        df['Slope (Degree)'] * df['Rainfall (mm)'] / (df['NDVI_Value'] + 0.1) *
        (1 + df['TRI']) * (1 - np.tanh(np.abs(df['TWI'])/10))  # Wetness penalty
    )
    
    # 6. FEATURES & SCALE
    features = [
        'Aspect (Degree)', 'Drainage Density (m)', 'Elevation (m)', 'NDVI_Value',
        'Plan Curvature', 'Profile Curvature', 'Rainfall (mm)', 'Slope (Degree)',
        'Soil Type_enc', 'Geology_Formation_enc', 'Land_Use_enc', 'SPI', 'TPI', 'TRI', 'TWI'
    ]
    X = df[features].copy()
    y = df['erosion_risk']
    
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features, index=X.index)
    
    # 7. SAVE
    processed_dir.mkdir(exist_ok=True)
    X_scaled.to_csv(processed_dir / 'X_processed.csv', index=False)
    y.to_csv(processed_dir / 'y_erosion_risk.csv', index=False)
    pd.concat([X.describe(), y.describe()], axis=1).to_csv(processed_dir / 'stats_summary.csv')
    
    joblib.dump({'encoders': encoders, 'scaler': scaler, 'imputer_num': imputer_num}, 
                processed_dir / 'artifacts.pkl')
    
    print(f"Final X: {X_scaled.shape}, y: {len(y)}, Target mean: {y.mean():.2f}")
    print("Done! Check data/processed/")
    return X_scaled, y

if __name__ == "__main__":
    raw_path = Path('../data/raw') / 'Merged_Woredas_All.xlsx'
    processed_dir = Path('../data/processed')
    preprocess_data(raw_path, processed_dir)
