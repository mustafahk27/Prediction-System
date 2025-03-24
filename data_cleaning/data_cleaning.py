import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def data_cleaning(data):
    """
    Clean and prepare the data, removing duplicates and handling column names.
    Ensures data is not empty and required columns are present.
    """
    if data is None:
        logger.error("Input data is None")
        return None
        
    if data.empty:
        logger.error("Input data is empty")
        return None
        
    logger.info(f"Data cleaning started with {len(data)} rows and {len(data.columns)} columns")
    
    # Check for required columns
    if 'Date' not in data.columns and not any(col.lower() == 'date' for col in data.columns):
        logger.error("Date column missing in input data")
        return None
        
    if 'close' not in data.columns and not any(col.lower() == 'close' for col in data.columns):
        logger.error("Close column missing in input data")
        return None
    
    # Make a copy to avoid modifying the original
    data = data.copy()
    
    # Handle missing values more carefully - only drop rows with missing values in critical columns
    critical_cols = ['Date', 'close']
    critical_cols = [col for col in critical_cols if col in data.columns]
    
    # Count rows before dropping
    rows_before = len(data)
    
    # Drop rows with missing values in critical columns only
    data = data.dropna(subset=critical_cols)
    
    # Count rows after dropping
    rows_after = len(data)
    if rows_before > rows_after:
        logger.warning(f"Dropped {rows_before - rows_after} rows with missing values in critical columns")
    
    # For non-critical columns, fill missing values instead of dropping
    for col in data.columns:
        if col not in critical_cols and data[col].isna().any():
            if data[col].dtype in [np.float64, np.int64]:
                # Fill numeric columns with median
                data[col] = data[col].fillna(data[col].median())
            else:
                # Fill non-numeric columns with mode
                data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else "Unknown")
    
    # Drop duplicate rows
    rows_before = len(data)
    data = data.drop_duplicates()
    rows_after = len(data)
    if rows_before > rows_after:
        logger.warning(f"Dropped {rows_before - rows_after} duplicate rows")
    
    # Fix duplicate column names
    # Rename duplicate 'volume' columns
    if 'volume' in data.columns and 'Volume' in data.columns:
        data = data.rename(columns={'volume': 'fear_greed_volume'})
    
    # Rename duplicate 'close' columns
    if 'close' in data.columns and 'Close' in data.columns:
        data = data.rename(columns={'close': 'fear_greed_close'})
    
    # Convert all column names to lowercase
    data.columns = data.columns.str.lower()
    
    # Remove any remaining duplicate columns
    data = data.loc[:, ~data.columns.duplicated()]
    
    # Ensure date column is properly named and formatted
    date_col = next((col for col in data.columns if col.lower() == 'date'), None)
    if date_col and date_col != 'date':
        data = data.rename(columns={date_col: 'date'})
    
    # Convert date to datetime
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
    
    # Final check for empty DataFrame
    if data.empty:
        logger.error("Data is empty after cleaning")
        return None
        
    # Check for sufficient rows after cleaning
    if len(data) < 30:
        logger.error(f"Data has only {len(data)} rows after cleaning, which is insufficient for training")
        return None
    
    logger.info(f"Data cleaning completed with {len(data)} rows and {len(data.columns)} columns")
    return data