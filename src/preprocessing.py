import pandas as pd
from sklearn.preprocessing import StandardScaler

def clean_and_feature_engineer(df):
    """
    Cleans raw manufacturing data and engineers temporal and geometry features.
    """
    # Drop rows with critical missing values
    df_clean = df.dropna(subset=['Feede_ID', 'Coordinate_X', 'Coordinate_Y', 'Designator']).copy()
    
    # A. Feature Engineering (Dates & Times)
    df_clean['NPM_Date'] = pd.to_datetime(df_clean['NPM_Date'], errors='coerce')
    # Use 'Verif_Date' if it exists, otherwise use 'Verification_Date'
    date_col = 'Verif_Date' if 'Verif_Date' in df_clean.columns else 'Verification_Date'
    
    if date_col in df_clean.columns:
        df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
        df_clean['Delay_to_Verif_Mins'] = (df_clean[date_col] - df_clean['NPM_Date']).dt.total_seconds() / 60.0
    else:
        df_clean['Delay_to_Verif_Mins'] = 0.0

    df_clean['NPM_Hour'] = df_clean['NPM_Date'].dt.hour
    df_clean['NPM_DayOfWeek'] = df_clean['NPM_Date'].dt.dayofweek
    
    # Fill missing engineered values with 0
    df_clean[['NPM_Hour', 'NPM_DayOfWeek', 'Delay_to_Verif_Mins']] = df_clean[['NPM_Hour', 'NPM_DayOfWeek', 'Delay_to_Verif_Mins']].fillna(0)
    
    return df_clean

def aggregate_to_boards(df_clean):
    """
    Aggregates component-level data into board-level summaries for machine learning.
    """
    # B. Target Logic (If any component failed, the whole board failed)
    # Ensure Componenet_Result is clean
    df_clean['Componenet_Result'] = df_clean['Componenet_Result'].astype(str).str.strip()
    
    # Map Pass -> 0, everything else (Fail, etc) -> 1
    # This ensures that 'Fail' maps to 1 for training.
    df_clean['y_Target'] = df_clean['Componenet_Result'].map(lambda x: 0 if x == 'Pass' else 1)

    # Initialize Scaler for coordinates
    scaler = StandardScaler()
    df_clean[['Coordinate_X_Scaled', 'Coordinate_Y_Scaled']] = scaler.fit_transform(df_clean[['Coordinate_X', 'Coordinate_Y']])

    df_boards = df_clean.groupby('Pattern_Barcode').agg(
        # TARGET: If any component failed, the whole board failed (max y_Target)
        Board_Failed=('y_Target', 'max'),
        
        # FEATURES
        Total_Components_Placed=('Designator', 'count'),
        Unique_Feeders_Used=('Feede_ID', 'nunique'),
        Avg_Coordinate_X=('Coordinate_X_Scaled', 'mean'),
        Avg_Coordinate_Y=('Coordinate_Y_Scaled', 'mean'),
        
        # Time features (first component's timestamp is representative for the board)
        NPM_Hour=('NPM_Hour', 'first'),
        NPM_DayOfWeek=('NPM_DayOfWeek', 'first'),
        Delay_to_Verif_Mins=('Delay_to_Verif_Mins', 'first')
    ).reset_index()
    
    return df_boards, scaler
