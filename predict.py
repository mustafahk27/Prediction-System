import torch
import pandas as pd
import numpy as np
from model_architecture.model_architecture import build_lstm_model, create_lstm_tensors
from DataPreprocessing.data_preprocessing import preprocess_data
from data_generator.crypto_pipeline import get_combined_data, get_combined_data_coingecko
from data_cleaning.data_cleaning import data_cleaning
from dotenv import load_dotenv
import os
import logging
import glob

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
WINDOW_SIZE = int(os.getenv('WINDOW_SIZE', '35'))  # Default to 35 if not set
USE_COINGECKO_ONLY = os.getenv('USE_COINGECKO_ONLY', 'True').lower() == 'true'

def get_latest_data_file(symbol):
    """
    Find the most recent data file for a given symbol.
    Looks for files in both Stored_data and crypto_data directories.
    Tries both Excel and CSV formats.
    """
    # Check for cleaned data files
    patterns = [
        f"Stored_data/{symbol}_cleaned_data.xlsx",
        f"Stored_data/{symbol}_cleaned_data.csv",
        f"Stored_data/{symbol.lower()}_cleaned_data.xlsx",
        f"Stored_data/{symbol.lower()}_cleaned_data.csv",
        f"crypto_data/{symbol}_combined_data_coingecko.csv",  # Check in crypto_data directory too
    ]
    
    # Check each pattern
    for pattern in patterns:
        if os.path.exists(pattern):
            logger.info(f"Found data file: {pattern}")
            return pattern
    
    # If no exact match, search for files with the symbol in the name
    all_files = glob.glob("Stored_data/*.csv") + glob.glob("Stored_data/*.xlsx") + glob.glob("crypto_data/*.csv")
    matching_files = [f for f in all_files if symbol.lower() in f.lower()]
    
    # Sort by modification time to get the newest file
    if matching_files:
        newest_file = max(matching_files, key=os.path.getmtime)
        logger.info(f"Found newest matching file: {newest_file}")
        return newest_file
    
    return None

def make_predictions(coin_id="bitcoin"):
    try:
        logger.info(f"Starting prediction process for {coin_id}")
        
        # Try to get latest data file
        file_path = get_latest_data_file(coin_id)
        
        if file_path:
            logger.info(f"Using existing data file: {file_path}")
            # Load the data based on file extension
            if file_path.endswith('.xlsx'):
                data = pd.read_excel(file_path)
                logger.info("Loaded data from Excel file")
            else:
                data = pd.read_csv(file_path)
                logger.info("Loaded data from CSV file")
                
            # Clean the data
            data = data_cleaning(data)
        else:
            logger.info(f"No existing data found for {coin_id}, fetching new data...")
            
            # Load the model checkpoint to check data source
            checkpoint_path = f"Stored_data/{coin_id}_lstm_model.pt"
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
                
            checkpoint = torch.load(
                checkpoint_path,
                map_location=torch.device('cpu')  # Load to CPU first
            )
            
            # Check data source from checkpoint
            data_source = checkpoint.get('data_source', 'mixed')  # Default to mixed if not specified
            
            # Use the appropriate data fetching function
            if data_source == 'coingecko' or USE_COINGECKO_ONLY:
                logger.info(f"Using CoinGecko-only data for {coin_id}")
                Raw_Data = get_combined_data_coingecko(coin_id)
            else:
                logger.info(f"Using mixed data sources for {coin_id}")
                Raw_Data = get_combined_data(coin_id)
                
            if Raw_Data is None:
                raise ValueError(f"Failed to fetch data for {coin_id}")
                
            # Clean the generated data
            data = data_cleaning(Raw_Data)
            logger.info(f"Generated and cleaned new data for {coin_id}")
        
        # Standardize date column
        if 'date' in data.columns:
            data.rename(columns={'date': 'Date'}, inplace=True)
        if 'Date' not in data.columns:
            # Try to find date column case-insensitively
            date_col = next((col for col in data.columns if col.lower() == 'date'), None)
            if date_col:
                data = data.rename(columns={date_col: 'Date'})
            else:
                raise ValueError("Date column not found in the data")
        
        data['Date'] = pd.to_datetime(data['Date'])
        dates = data['Date']
        
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Date range: {data['Date'].min()} to {data['Date'].max()}")
        logger.info(f"Columns in data: {data.columns.tolist()}")
        
        # Find close column case-insensitively
        close_col = next((col for col in data.columns if col.lower() == 'close'), None)
        if not close_col:
            raise ValueError("Required 'close' column not found in the data")
            
        # Store original close prices before preprocessing
        original_close_prices = data[close_col].values.copy()
        logger.info(f"Original close price range: [{original_close_prices.min():.2f}, {original_close_prices.max():.2f}]")
        
        # Load the saved model checkpoint
        checkpoint_path = f"Stored_data/{coin_id}_lstm_model.pt"
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}")
            
        checkpoint = torch.load(
            checkpoint_path,
            map_location=torch.device('cpu')  # Load to CPU first
        )
        
        # Get the feature names used during training
        if 'feature_names' not in checkpoint:
            raise ValueError("Feature names not found in checkpoint. Please retrain the model.")
        training_features = checkpoint['feature_names']
        logger.info(f"Number of features in trained model: {len(training_features)}")
        
        # Get feature scaler from checkpoint
        feature_scaler = checkpoint.get('feature_scaler')
        target_scaler = checkpoint.get('target_scaler')
        
        if not feature_scaler or not target_scaler:
            raise ValueError("Scalers not found in checkpoint. Please retrain the model.")
        
        # IMPORTANT: Separate target before preprocessing to avoid double scaling
        logger.info("Separating 'close' column from features to avoid double scaling")
        target = data[close_col]
        # Make sure to remove both date and close from features
        features_df = data.drop(['Date', close_col], axis=1)
        
        # Verify that close is not in features_df (case-insensitive)
        close_cols = [col for col in features_df.columns if col.lower() == 'close']
        if close_cols:
            logger.warning("'close' column still found in features after removal!")
            features_df = features_df.drop(close_cols, axis=1)
            
        # Preprocess features using the saved scaler
        logger.info("Preprocessing features using saved scaler")
        features_df.columns = features_df.columns.str.lower()
        expected_cols = feature_scaler.feature_names_in_
        
        # Add missing columns as zeros
        for col in expected_cols:
            if col not in features_df.columns:
                logger.info(f"Adding missing column: {col}")
                features_df[col] = 0
                
        # Select only the expected columns in the correct order
        features_df = features_df[expected_cols]
        features_scaled = feature_scaler.transform(features_df)
        features_for_lstm = pd.DataFrame(features_scaled, columns=expected_cols, index=features_df.index)
        
        logger.info(f"Features shape after preprocessing: {features_for_lstm.shape}")
        
        # Create LSTM tensors
        lstm_input = create_lstm_tensors(features_for_lstm, WINDOW_SIZE)
        logger.info(f"LSTM input shape: {lstm_input.shape}")
        
        # Initialize model with correct input dimension
        input_size = checkpoint.get('input_size')
        if not input_size:
            input_size = lstm_input.shape[2]
            
        model = build_lstm_model(lstm_input)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Get the last window of data
        current_window = torch.FloatTensor(lstm_input[-1:]).to(device)
        
        # Generate predictions for next 30 days
        predictions = []
        prediction_dates = []
        last_date = dates.iloc[-1]
        
        logger.info("Starting prediction loop for next 30 days")
        
        with torch.no_grad():
            for i in range(30):
                # Make prediction
                pred = model(current_window)
                if isinstance(pred, tuple):
                    pred = pred[0]
                pred_numpy = pred.cpu().numpy().reshape(-1, 1)
                pred_unscaled = target_scaler.inverse_transform(pred_numpy).flatten()[0]
                
                # Store prediction
                next_date = last_date + pd.Timedelta(days=i+1)
                predictions.append(float(pred_unscaled))
                prediction_dates.append(next_date.strftime("%Y-%m-%d"))
                
                if i < 29:  # Don't need to update for the last prediction
                    # Update the window - shift everything left and add new prediction
                    # First, scale the new prediction
                    new_pred_scaled = target_scaler.transform([[pred_unscaled]])[0, 0]
                    
                    # Update features for next prediction (simple forward fill of last known features)
                    next_features = features_for_lstm.iloc[-1:].values
                    
                    # Create new window by shifting and adding new prediction
                    new_window = current_window.cpu().numpy().copy()
                    new_window[0] = np.roll(new_window[0], -1, axis=0)  # Shift left
                    new_window[0, -1] = next_features[0]  # Add new features
                    current_window = torch.FloatTensor(new_window).to(device)
        
        # Create prediction DataFrame
        prediction_df = pd.DataFrame({
            'Date': prediction_dates,
            'Predicted_Price': predictions
        })
        
        # Calculate statistics
        last_known_price = original_close_prices[-1]
        first_day_change = ((predictions[0] - last_known_price) / last_known_price) * 100
        avg_price = np.mean(predictions)
        min_price = np.min(predictions)
        max_price = np.max(predictions)
        
        # Print detailed results
        logger.info("\nPrediction Summary:")
        logger.info(f"Last known date: {dates.iloc[-1].strftime('%Y-%m-%d')}")
        logger.info(f"Last known price: ${last_known_price:,.2f}")
        logger.info(f"First day prediction: ${predictions[0]:,.2f} ({first_day_change:+.2f}%)")
        logger.info(f"30-day prediction range: ${min_price:,.2f} to ${max_price:,.2f}")
        logger.info(f"30-day average predicted price: ${avg_price:,.2f}")
        
        # Save predictions with more details
        prediction_df['Price_Change'] = prediction_df['Predicted_Price'].pct_change()
        prediction_df['Price_Change'].iloc[0] = first_day_change / 100
        prediction_df['Cumulative_Change'] = ((prediction_df['Predicted_Price'] - last_known_price) / last_known_price) * 100
        
        # Save to both Excel and CSV
        excel_path = f"Stored_data/{coin_id}_latest_predictions.xlsx"
        csv_path = f"Stored_data/{coin_id}_latest_predictions.csv"
        prediction_df.to_excel(excel_path, index=False)
        prediction_df.to_csv(csv_path, index=False)
        
        logger.info(f"\nPredictions saved to:")
        logger.info(f"- Excel: {excel_path}")
        logger.info(f"- CSV: {csv_path}")
        
        return prediction_df
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    try:
        coin_id = os.getenv('CRYPTO_SYMBOL', 'SOLUSDT')
        predictions = make_predictions(coin_id)
        print(f"\nPrediction process completed successfully for {coin_id}!")
    except Exception as e:
        print(f"Error in main: {str(e)}")
