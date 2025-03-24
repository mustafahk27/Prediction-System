from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import pandas as pd
import os
from training import train_crypto_model
from model_architecture.model_architecture import build_lstm_model, LSTMAttention, create_lstm_tensors
from DataPreprocessing.data_preprocessing import preprocess_data
from data_generator.crypto_pipeline import get_combined_data, get_combined_data_coingecko
from data_cleaning.data_cleaning import data_cleaning
import numpy as np
import traceback
import glob
from typing import List, Optional
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
from bs4 import BeautifulSoup
from hyperbrowser import Hyperbrowser
from hyperbrowser.models import ScrapeOptions, StartScrapeJobParams, CreateSessionParams
import scraping_predictions

# Load environment variables
load_dotenv()

# Get environment variables
WINDOW_SIZE = int(os.getenv('WINDOW_SIZE', '35'))
USE_COINGECKO_ONLY = os.getenv('USE_COINGECKO_ONLY', 'True').lower() == 'true'

# Helper function for safe feature transformation
def safe_transform_features(scaler, features_df, expected_feature_names=None):
    """
    Safely transform features using a fitted scaler, handling potential dimension mismatches.
    """
    try:
        print(f"Features shape: {features_df.shape}")
        print(f"Features columns: {features_df.columns.tolist()}")
        print(f"Scaler n_features_in_: {scaler.n_features_in_}")
        
        # If we have expected feature names and they don't match, try to fix
        if expected_feature_names and len(expected_feature_names) == scaler.n_features_in_:
            # Add missing columns as zeros
            for col in expected_feature_names:
                if col not in features_df.columns:
                    print(f"Adding missing column: {col}")
                    features_df[col] = 0
            
            # Reorder to match expected order
            features_df = features_df[expected_feature_names]
        
        # Final check before transformation
        if features_df.shape[1] != scaler.n_features_in_:
            raise ValueError(f"Feature count mismatch! Expected {scaler.n_features_in_}, got {features_df.shape[1]}")
        
        # Transform the features
        scaled_features = scaler.transform(features_df)
        return pd.DataFrame(
            scaled_features,
            columns=features_df.columns,
            index=features_df.index
        )
    except Exception as e:
        print(f"Error in safe_transform_features: {str(e)}")
        traceback.print_exc()
        raise e

# Helper function to get the latest data file for a symbol
def get_latest_data_file(coin_id):
    """
    Find the most recent data file for a given coin_id
    """
    # Updated patterns to use coin_id instead of symbol
    patterns = [
        f"Stored_data/{coin_id}_cleaned_data.xlsx",
        f"Stored_data/{coin_id}_cleaned_data.csv",
        f"crypto_data/{coin_id}_combined_data_coingecko.csv",
    ]
    
    # Check each pattern
    for pattern in patterns:
        if os.path.exists(pattern):
            print(f"Found data file: {pattern}")
            return pattern
    
    # If no exact match, search for files with the coin_id in the name
    all_files = glob.glob("Stored_data/*.csv") + glob.glob("Stored_data/*.xlsx") + glob.glob("crypto_data/*.csv")
    matching_files = [f for f in all_files if coin_id.lower() in f.lower()]
    
    # Sort by modification time to get the newest file
    if matching_files:
        newest_file = max(matching_files, key=os.path.getmtime)
        print(f"Found newest matching file: {newest_file}")
        return newest_file
    
    return None

# Initialize FastAPI app
app = FastAPI(title="Cryptocurrency Price Prediction API")

# Configure CORS - Update this to match your frontend URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

class CryptoRequest(BaseModel):
    coin_id: str
    use_coingecko_only: Optional[bool] = True

@app.post("/train")
async def train_model(request: CryptoRequest):
    """
    Train a model for a specific cryptocurrency
    
    Args:
        request: CryptoRequest containing the coin_id (e.g., "BTCUSDT")
    """
    try:
        # Get raw coin_id without modifications
        coin_id = request.coin_id.lower().strip()
        
        print(f"Starting training for {coin_id}")
        
        # Set environment variable for CoinGecko usage
        os.environ['USE_COINGECKO_ONLY'] = str(request.use_coingecko_only)
        
        # Train the model
        model, scaler, target_scaler = train_crypto_model(coin_id)
        
        # Return success response with training metrics
        return {
            "status": "success",
            "message": f"Model trained successfully for {coin_id}",
            "model_path": f"Stored_data/{coin_id}_lstm_model.pt",
            "predictions_path": f"Stored_data/{coin_id}_predictions.xlsx",
            "training_plots": f"Stored_data/{coin_id}_training_results.png",
            "data_source": "CoinGecko only" if request.use_coingecko_only else "Mixed sources"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error training model for {request.coin_id}: {str(e)}"
        )

@app.post("/predict")
async def predict(request: CryptoRequest):
    try:
        coin_id = request.coin_id.lower().strip()
        
        # Load model and checkpoint
        model_path = f"Stored_data/{coin_id}_lstm_model.pt"
        if not os.path.exists(model_path):
            # Check if this is a typo - try to find similar symbols
            all_models = glob.glob("Stored_data/*_lstm_model.pt")
            model_symbols = [os.path.basename(m).split('_')[0] for m in all_models]
            
            if model_symbols:
                suggestion = f"Available models: {', '.join(model_symbols)}"
                raise HTTPException(
                    status_code=404, 
                    detail=f"No trained model found for {coin_id}. {suggestion}"
                )
            else:
                raise HTTPException(
                    status_code=404, 
                    detail=f"No trained model found for {coin_id}. Please train a model first using the /train endpoint."
                )
            
        # Load checkpoint
        checkpoint = torch.load(model_path)
        print("Checkpoint keys:", checkpoint.keys())
        
        # Get model parameters
        input_size = checkpoint.get('input_size')
        feature_scaler = checkpoint.get('feature_scaler')
        target_scaler = checkpoint.get('target_scaler')
        data_source = checkpoint.get('data_source', 'mixed')  # Default to mixed if not specified
        
        if not all([input_size, feature_scaler, target_scaler]):
            raise HTTPException(status_code=500, detail="Model checkpoint missing required components")
            
        # Try to get latest data or fetch new data
        file_path = get_latest_data_file(coin_id)
        if not file_path:
            print(f"No existing data found for {coin_id}, fetching new data...")
            
            # Use the appropriate data fetching function based on the model's training data source
            if data_source == 'coingecko' or request.use_coingecko_only:
                print(f"Using CoinGecko-only data for {coin_id}")
                Raw_Data = get_combined_data_coingecko(coin_id)
            else:
                print(f"Using mixed data sources for {coin_id}")
                Raw_Data = get_combined_data(coin_id)
                
            if Raw_Data is None or Raw_Data.empty:
                raise HTTPException(status_code=500, detail=f"Failed to fetch data for {coin_id} or data is empty")
            # Clean the generated data
            data = data_cleaning(Raw_Data)
            if data is None or data.empty:
                raise HTTPException(status_code=500, detail=f"Data cleaning resulted in empty DataFrame for {coin_id}")
        else:
            try:
                data = pd.read_csv(file_path)
            except Exception as e:
                print(f"Could not load CSV: {str(e)}, trying Excel")
                try:
                    data = pd.read_excel(file_path)
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"No data found for {coin_id}")
        
        # Clean the loaded data
        data = data_cleaning(data)
        if data is None or data.empty:
            raise HTTPException(status_code=500, detail=f"Data cleaning resulted in empty DataFrame for {coin_id}")
        
        print(f"Data shape after cleaning: {data.shape}")
        print("Columns after cleaning:", data.columns.tolist())
        
        # Find date and close columns case-insensitively
        date_col = next((col for col in data.columns if col.lower() == 'date'), None)
        close_col = next((col for col in data.columns if col.lower() == 'close'), None)
        
        if not date_col or not close_col:
            raise HTTPException(status_code=500, detail=f"Required columns 'Date' and/or 'close' not found in the data for {coin_id}")
        
        # Store dates before preprocessing
        dates = data[date_col]
        
        # IMPORTANT: Separate target before preprocessing to avoid double scaling
        print("Separating 'close' column from features to avoid double scaling")
        target = data[close_col]
        # Make sure to remove both date and close from features
        features_df = data.drop([date_col, close_col], axis=1)
        
        # Verify that close is not in features_df (case-insensitive)
        close_cols = [col for col in features_df.columns if col.lower() == 'close']
        if close_cols:
            print("ERROR: 'close' column still found in features after removal!")
            features_df = features_df.drop(close_cols, axis=1)
            
        # Preprocess features using the saved scaler
        print("Preprocessing features using saved scaler")
        features_df.columns = features_df.columns.str.lower()
        expected_cols = feature_scaler.feature_names_in_
        
        # Add missing columns as zeros
        for col in expected_cols:
            if col not in features_df.columns:
                print(f"Adding missing column: {col}")
                features_df[col] = 0
                
        # Select only the expected columns in the correct order
        features_df = features_df[expected_cols]
        features_scaled = feature_scaler.transform(features_df)
        features_for_lstm = pd.DataFrame(features_scaled, columns=expected_cols, index=features_df.index)
        
        print(f"Features shape after preprocessing: {features_for_lstm.shape}")
        
        # Create LSTM tensors
        try:
            # Import the function directly from model_architecture if needed
            from model_architecture.model_architecture import create_lstm_tensors
            lstm_input = create_lstm_tensors(features_for_lstm, WINDOW_SIZE)
            print("LSTM input shape:", lstm_input.shape)
        except Exception as e:
            print(f"Error creating LSTM tensors: {str(e)}")
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Error creating LSTM tensors: {str(e)}")
        
        # Check if we have enough data
        if len(lstm_input) < 1:
            raise HTTPException(status_code=500, detail=f"Insufficient data for prediction after preprocessing for {coin_id}")
        
        # Set device for PyTorch
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Initialize model with correct input dimension
        model = LSTMAttention(
            input_dim=input_size,
            window_size=WINDOW_SIZE
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)  # Move model to the device
        model.eval()
        
        # Predict
        predictions = []
        prediction_dates = []
        last_date = dates.iloc[-1]
        
        # Get the last window of data
        current_window = lstm_input[-1:]  # Get the last prepared window
        current_tensor = torch.FloatTensor(current_window).to(device)  # Move tensor to the device
        
        for i in range(30):
            try:
                # Make prediction
                with torch.no_grad():
                    pred = model(current_tensor)
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
                    # FIX: Use numpy's copy method for tensors
                    new_window = current_tensor.cpu().numpy().copy()
                    new_window[0] = np.roll(new_window[0], -1, axis=0)  # Shift left
                    new_window[0, -1] = next_features[0]  # Add new features
                    current_tensor = torch.FloatTensor(new_window).to(device)
                
            except Exception as e:
                print(f"Error during prediction iteration {i}: {str(e)}")
                traceback.print_exc()
                if predictions:
                    break
                raise HTTPException(status_code=500, detail=f"Error during prediction calculation: {str(e)}")
        
        # Save and return results
        prediction_df = pd.DataFrame({
            'Date': prediction_dates,
            'Predicted_Price': predictions
        })
        
        # Calculate additional metrics
        # FIX: Handle the case when we only have one prediction
        first_day_change = 0
        if len(predictions) > 1:
            prediction_df['Daily_Change'] = prediction_df['Predicted_Price'].pct_change() * 100
            prediction_df['Daily_Change'].iloc[0] = ((predictions[0] - target.iloc[-1]) / target.iloc[-1]) * 100
            first_day_change = prediction_df['Daily_Change'].iloc[0]
        else:
            # If we only have one prediction, calculate the change from the last known price
            first_day_change = ((predictions[0] - target.iloc[-1]) / target.iloc[-1]) * 100
            prediction_df['Daily_Change'] = [first_day_change]  # Add as a single value
        
        # Calculate cumulative change from last known price
        last_actual_price = target.iloc[-1]
        prediction_df['Cumulative_Change'] = ((prediction_df['Predicted_Price'] - last_actual_price) / last_actual_price) * 100
        
        # Save to Excel
        excel_path = f"Stored_data/{coin_id}_latest_predictions.xlsx"
        prediction_df.to_excel(excel_path, index=False)
        print(f"Predictions saved to {excel_path}")
        
        # Calculate summary statistics
        max_price = prediction_df['Predicted_Price'].max()
        min_price = prediction_df['Predicted_Price'].min()
        avg_price = prediction_df['Predicted_Price'].mean()
        max_change = prediction_df['Cumulative_Change'].max()
        min_change = prediction_df['Cumulative_Change'].min()
        
        # Create detailed prediction data for response
        prediction_data = [
            {
                "date": date,
                "predicted_price": float(price),
                "day": i + 1,
                "daily_change_percent": float(prediction_df['Daily_Change'].iloc[i]),
                "cumulative_change_percent": float(prediction_df['Cumulative_Change'].iloc[i])
            }
            for i, (date, price) in enumerate(zip(prediction_dates, predictions))
        ]
        
        # Create response with detailed information
        response = {
            "status": "success",
            "coin_id": coin_id,
            "last_date": last_date.strftime("%Y-%m-%d"),
            "last_price": float(last_actual_price),
            "prediction_summary": {
                "initial_price_change_percent": float(first_day_change),
                "max_predicted_price": float(max_price),
                "min_predicted_price": float(min_price),
                "avg_predicted_price": float(avg_price),
                "max_cumulative_change_percent": float(max_change),
                "min_cumulative_change_percent": float(min_change),
                "prediction_period": f"{len(predictions)} days",
                "data_source": data_source
            },
            "predictions": prediction_data
        }
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error making prediction for {request.coin_id}: {str(e)}"
        )

@app.post("/predict_new")
async def predict_new(request: CryptoRequest):
    """
    Get 30-day price predictions for a cryptocurrency from digitalcoinprice.com
    
    Args:
        request: CryptoRequest containing the coin_id
    Returns:
        JSON with 30 days of price predictions
    """
    try:
        # Get the coin name from the request
        coin_name = request.coin_id.strip().lower()
        
        # Call the scraping function
        prediction_data = await scraping_predictions.get_coin_predictions(coin_name)
        
        return prediction_data
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting predictions for {request.coin_id}: {str(e)}"
        )

@app.get("/")
async def root():
    return {"message": "Cryptocurrency Price Prediction API using CoinGecko data"}

if __name__ == "__main__":
    import uvicorn
    # Update host to 0.0.0.0 to make it accessible externally
    uvicorn.run(app, host="0.0.0.0", port=8000) 
