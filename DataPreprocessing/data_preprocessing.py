import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.exceptions import NotFittedError
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def preprocess_data(df):
    """Preprocess data with enterprise-grade robustness and version compatibility."""
    try:
        # Check for empty DataFrame
        if df is None:
            logger.error("Input DataFrame is None")
            # Return a dummy scaler and empty DataFrame to avoid errors
            dummy_scaler = MinMaxScaler(feature_range=(-1, 1))
            dummy_scaler.fit(np.array([[0], [1]]))  # Fit with dummy data
            return pd.DataFrame(), dummy_scaler
            
        if df.empty:
            logger.error("Input DataFrame is empty")
            # Return a dummy scaler and empty DataFrame to avoid errors
            dummy_scaler = MinMaxScaler(feature_range=(-1, 1))
            dummy_scaler.fit(np.array([[0], [1]]))  # Fit with dummy data
            return pd.DataFrame(), dummy_scaler
            
        # Check for minimum rows
        if len(df) < 2:
            logger.error(f"Input DataFrame has only {len(df)} rows, minimum 2 required")
            # Return a dummy scaler and empty DataFrame to avoid errors
            dummy_scaler = MinMaxScaler(feature_range=(-1, 1))
            dummy_scaler.fit(np.array([[0], [1]]))  # Fit with dummy data
            return pd.DataFrame(), dummy_scaler
        
        logger.info("Input DataFrame info:")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"Columns before preprocessing: {df.columns.tolist()}")
        
        # Ensure no duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Convert all column names to lowercase
        df.columns = df.columns.str.lower()
        
        # Handle the 'classification' column from Fear & Greed Index specially
        if 'classification' in df.columns:
            # Convert classification to numeric values
            classification_map = {
                'Extreme Fear': 0,
                'Fear': 0.25,
                'Neutral': 0.5,
                'Greed': 0.75,
                'Extreme Greed': 1.0
            }
            df['fear_greed_numeric'] = df['classification'].map(classification_map)
            df = df.drop('classification', axis=1)
        
        # Identify numeric columns (excluding any remaining categorical)
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        
        # Check if we have any numeric columns
        if len(numeric_cols) == 0:
            logger.error("No numeric columns found in the DataFrame")
            # Return a dummy scaler and empty DataFrame to avoid errors
            dummy_scaler = MinMaxScaler(feature_range=(-1, 1))
            dummy_scaler.fit(np.array([[0], [1]]))  # Fit with dummy data
            return pd.DataFrame(), dummy_scaler
        
        # Log a warning if 'close' is unexpectedly found in the features
        if 'close' in numeric_cols:
            logger.warning("'close' column found in features even though it should have been removed!")
            numeric_cols = numeric_cols.difference(['close'])
            if 'close' in df.columns:
                df = df.drop('close', axis=1)
        
        # Fill missing values in numeric columns
        for col in numeric_cols:
            if df[col].isna().any():
                logger.warning(f"Filling missing values in column '{col}'")
                df[col] = df[col].fillna(df[col].median())
        
        # Create preprocessing pipeline for numeric data
        numeric_transformer = Pipeline(steps=[
            ('scaler', MinMaxScaler(feature_range=(-1, 1)))
        ])
        
        # Create column transformer
        preprocessor = ColumnTransformer(
            transformers=[('num', numeric_transformer, numeric_cols)]
        )
        
        # Transform the data
        df_transformed = preprocessor.fit_transform(df)
        
        # Verify scaler parameters
        num_scaler = preprocessor.named_transformers_['num'].named_steps['scaler']
        if num_scaler.feature_range != (-1, 1):
            raise ValueError(f"Scaler feature range mismatch. Expected (-1, 1), got {num_scaler.feature_range}")
        
        # Create DataFrame with transformed data
        transformed_df = pd.DataFrame(
            df_transformed,
            columns=numeric_cols,
            index=df.index
        )
        
        logger.info(f"Columns after preprocessing: {transformed_df.columns.tolist()}")
        logger.info(f"Number of features after preprocessing: {transformed_df.shape[1]}")
        return transformed_df, preprocessor.named_transformers_['num'].named_steps['scaler']
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {str(e)}")
        logger.error(f"Input DataFrame shape: {df.shape if df is not None else 'None'}")
        
        # Return a dummy scaler and empty DataFrame to avoid errors
        dummy_scaler = MinMaxScaler(feature_range=(-1, 1))
        dummy_scaler.fit(np.array([[0], [1]]))  # Fit with dummy data
        return pd.DataFrame(), dummy_scaler

def add_gaussian_noise(df, columns_to_augment, noise_fraction=0.02):
    """
    Adds Gaussian noise to specified numerical columns in the DataFrame.

    Parameters:
    - df: The input DataFrame.
    - columns_to_augment: List of column names to which noise will be added.
    - noise_fraction: Fraction of the mean to determine the standard deviation of the noise.

    Returns:
    - DataFrame with added Gaussian noise in specified columns.
    """
    df_augmented = df.copy()  # Create a copy to avoid modifying the original DataFrame
    
    # Replace inf/-inf with NaN first
    df_augmented.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Check for NaN values before augmentation
    print("\n=== NaN values before augmentation ===")
    nan_counts = df_augmented.isnull().sum()
    print(nan_counts[nan_counts > 0])
    print(f"Total NaN values: {df_augmented.isnull().sum().sum()}")
    
    # Fill NaN values before augmentation to prevent issues
    if df_augmented.isnull().sum().sum() > 0:
        print("Filling NaN values before augmentation")
        for col in df_augmented.columns:
            if df_augmented[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df_augmented[col]):
                    # For numeric columns, use mean if possible
                    if df_augmented[col].count() > 0:  # If we have some non-NaN values
                        mean_val = df_augmented[col].mean()
                        if pd.isna(mean_val):  # If mean is NaN (all values might be NaN)
                            df_augmented[col] = df_augmented[col].fillna(0)
                            print(f"  - Filled NaNs in {col} with 0 (mean was NaN)")
                        else:
                            df_augmented[col] = df_augmented[col].fillna(mean_val)
                            print(f"  - Filled NaNs in {col} with mean: {mean_val}")
                    else:
                        df_augmented[col] = df_augmented[col].fillna(0)
                        print(f"  - Filled NaNs in {col} with 0 (all values were NaN)")
                else:
                    # For non-numeric columns, use most frequent value or 'unknown'
                    if df_augmented[col].count() > 0:  # If we have some non-NaN values
                        mode_val = df_augmented[col].mode()[0]
                        df_augmented[col] = df_augmented[col].fillna(mode_val)
                        print(f"  - Filled NaNs in {col} with mode: {mode_val}")
                    else:
                        df_augmented[col] = df_augmented[col].fillna('unknown')
                        print(f"  - Filled NaNs in {col} with 'unknown' (all values were NaN)")
    
    # Verify no NaNs remain before augmentation
    if df_augmented.isnull().sum().sum() > 0:
        print("WARNING: Still have NaNs after initial filling. Applying more aggressive filling.")
        df_augmented = df_augmented.fillna(0)  # Fill any remaining NaNs with 0
    
    count = 0
    for column in columns_to_augment:
        if column in df_augmented.columns:
            # Ensure column is numeric
            if not pd.api.types.is_numeric_dtype(df_augmented[column]):
                print(f"Skipping non-numeric column: {column}")
                continue
                
            # Calculate mean safely
            mean = df_augmented[column].mean()
            if pd.isna(mean) or mean == 0:
                print(f"Skipping column {column} due to NaN or zero mean")
                continue
                
            # Calculate standard deviation as a fraction of the mean
            std_dev = abs(noise_fraction * mean)  # Use absolute value to ensure positive std_dev
            
            # Generate Gaussian noise
            noise = np.random.normal(0, std_dev, size=df_augmented[column].shape)
            
            # Ensure noise doesn't contain NaN or inf
            if np.isnan(noise).any() or np.isinf(noise).any():
                print(f"WARNING: Generated noise for {column} contains NaN or inf values. Fixing...")
                noise = np.nan_to_num(noise, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Add noise to the column
            original_values = df_augmented[column].copy()
            df_augmented[column] += noise
            
            # Check if adding noise introduced NaNs or infs
            if df_augmented[column].isna().any() or np.isinf(df_augmented[column].to_numpy()).any():
                print(f"WARNING: Adding noise to {column} introduced NaN or inf values. Reverting to original values.")
                df_augmented[column] = original_values
            else:
                count += 1
                print(f"Added noise to column: {column} (mean: {mean}, std_dev: {std_dev})")

    print(f"Successfully augmented {count} columns")
    
    # Check for NaN values after augmentation
    print("\n=== NaN values after augmentation ===")
    nan_counts = df_augmented.isnull().sum()
    print(nan_counts[nan_counts > 0])
    print(f"Total NaN values: {df_augmented.isnull().sum().sum()}")
    
    # Fill any NaNs that might have been introduced during augmentation
    if df_augmented.isnull().sum().sum() > 0:
        print("Filling NaN values introduced during augmentation")
        df_augmented = df_augmented.fillna(df.fillna(method='ffill').fillna(method='bfill').fillna(0))
    
    # Final check for inf values
    if np.isinf(df_augmented.to_numpy()).any():
        print("WARNING: Inf values found after augmentation. Replacing with large finite values.")
        df_augmented = df_augmented.replace([np.inf, -np.inf], [1e10, -1e10])
    
    # Final verification
    if df_augmented.isnull().sum().sum() > 0:
        print("CRITICAL WARNING: Still have NaNs after all processing. Replacing with zeros as last resort.")
        df_augmented = df_augmented.fillna(0)
    
    return df_augmented

def data_augmentation(data):
    """
    Apply data augmentation by adding Gaussian noise to selected columns.
    
    Parameters:
    - data: The input DataFrame to augment
    
    Returns:
    - Augmented DataFrame
    """
    print("\n=== Starting Data Augmentation ===")
    
    # Make a copy to avoid modifying the original
    data = data.copy()
    
    # Define columns to augment - only include columns that exist in the data
    potential_columns = [
        'open', 'high', 'low', 'close', 'volume', 'quote_asset_volume',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
        'rsi', 'inflation_rate', 'avg_block_size',
        'num_transactions', 'miners_revenue'
    ]
    
    # Filter to only include columns that exist in the data (case-insensitive)
    columns_to_augment = []
    for col in potential_columns:
        # Try exact match first
        if col in data.columns:
            columns_to_augment.append(col)
        else:
            # Try case-insensitive match
            matches = [c for c in data.columns if c.lower() == col.lower()]
            if matches:
                columns_to_augment.append(matches[0])
    
    print(f"Columns selected for augmentation: {columns_to_augment}")
    
    # Apply augmentation
    data_augmented = add_gaussian_noise(data, columns_to_augment)
    
    print("=== Data Augmentation Complete ===\n")
    return data_augmented

def standardize_date_column(df):
    """
    Standardize the date column in a DataFrame.
    
    Args:
        df (pd.DataFrame): Input DataFrame with a date column
        
    Returns:
        pd.DataFrame: DataFrame with standardized date column
    """
    df = df.copy()
    
    # Find the date column
    date_col = None
    for col in df.columns:
        if col.lower() in ['date', 'timestamp', 'time']:
            date_col = col
            break
            
    # If found, standardize it
    if date_col:
        # Rename to 'Date' if not already
        if date_col != 'Date':
            df = df.rename(columns={date_col: 'Date'})
            
        # Convert to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Sort by date
        df = df.sort_values('Date')
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['Date'])
        
    return df
