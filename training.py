import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_cleaning.data_cleaning import data_cleaning
from data_generator.crypto_pipeline import get_combined_data, get_combined_data_coingecko
from dotenv import load_dotenv
import os
from DataPreprocessing.data_preprocessing import preprocess_data
from model_architecture.model_architecture import create_lstm_tensors, build_lstm_model  
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn

# Load environment variables
load_dotenv()

# Get environment variables
GENERATE_NEW_DATA = os.getenv('GENERATE_NEW_DATA', 'True').lower() == 'true'
WINDOW_SIZE = int(os.getenv('WINDOW_SIZE', '35'))
USE_COINGECKO_ONLY = os.getenv('USE_COINGECKO_ONLY', 'True').lower() == 'true'

def train_crypto_model(coin_id):
    """Train model for a specific cryptocurrency using CoinGecko data"""
    
    try:
        # Define the paths
        stored_data_dir = 'Stored_data'
        
        # Create directories if they don't exist
        os.makedirs(stored_data_dir, exist_ok=True)
        os.makedirs('crypto_data', exist_ok=True)
        
        if GENERATE_NEW_DATA:
            # Generate new data using the CoinGecko-oriented implementation
            print(f"Fetching data for {coin_id} from CoinGecko...")
            if USE_COINGECKO_ONLY:
                # Use exclusively CoinGecko data
                Raw_Data = get_combined_data_coingecko(coin_id)
            else:
                # Use the standard function (which now calls get_combined_data_coingecko internally)
                Raw_Data = get_combined_data(coin_id)
                
            if Raw_Data is None or Raw_Data.empty:
                raise ValueError(f"Failed to fetch data for {coin_id} or data is empty")
                
            print(f"Generated Raw Data for {coin_id}")
            print(f"Data shape: {Raw_Data.shape}")
            print(f"Date range: {Raw_Data['Date'].min()} to {Raw_Data['Date'].max()}")
            print(f"Columns: {Raw_Data.columns.tolist()}")
            print(Raw_Data.head())
            
            # Clean the generated data
            data = data_cleaning(Raw_Data)
            if data is None or data.empty:
                raise ValueError(f"Data cleaning resulted in empty DataFrame for {coin_id}")
                
            print("\nColumns after cleaning:")
            print(data.columns.tolist())
        else:
            # Try loading from CSV first, then Excel
            data_path_csv = f'stored_data/{coin_id}_data.csv'
            data_path_excel = f'stored_data/{coin_id}_data.xlsx'
            
            try:
                data = pd.read_csv(data_path_csv)
            except Exception as e:
                print(f"Could not load CSV: {str(e)}, trying Excel")
                try:
                    data = pd.read_excel(data_path_excel)
                except Exception as e:
                    raise ValueError(f"No data found for {coin_id}")
            
            if data is None or data.empty:
                raise ValueError(f"Loaded data is empty for {coin_id}")
                
            print(f"Loaded data for {coin_id}")
            
            # Ensure Date column is properly formatted
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date'])
            else:
                # Try to find date column case-insensitively
                date_col = next((col for col in data.columns if col.lower() == 'date'), None)
                if date_col:
                    data = data.rename(columns={date_col: 'Date'})
                    data['Date'] = pd.to_datetime(data['Date'])
                else:
                    raise ValueError("Date column not found in the data")

        # Check if we have enough data
        if len(data) < WINDOW_SIZE + 30:  # Need at least window size + some data for testing
            raise ValueError(f"Insufficient data: only {len(data)} rows. Need at least {WINDOW_SIZE + 30} rows.")

        # Save the raw cleaned data in both formats
        cleaned_data_path_excel = os.path.join(stored_data_dir, f'{coin_id}_cleaned_data.xlsx')
        cleaned_data_path_csv = os.path.join(stored_data_dir, f'{coin_id}_cleaned_data.csv')
        
        try:
            data.to_excel(cleaned_data_path_excel, index=False)
        except Exception as e:
            print(f"Warning: Could not save to Excel ({str(e)})")
        
        data.to_csv(cleaned_data_path_csv, index=False)
        print(f"Cleaned data saved to {cleaned_data_path_csv}")

        # Find date and close columns case-insensitively
        date_col = next((col for col in data.columns if col.lower() == 'date'), None)
        close_col = next((col for col in data.columns if col.lower() == 'close'), None)
        
        if not date_col or not close_col:
            raise ValueError("Required columns 'Date' and/or 'close' not found in the data")
        
        # Store dates before preprocessing
        dates = data[date_col]
        
        # IMPORTANT: Separate target before preprocessing to avoid double scaling
        print("Separating 'close' column from features to avoid double scaling")
        target = data[close_col]
        
        # Check if target has enough non-NaN values
        if target.isna().sum() > len(target) * 0.1:  # If more than 10% are NaN
            print(f"Warning: Target has {target.isna().sum()} NaN values out of {len(target)}")
            # Fill NaN values in target
            target = target.fillna(method='ffill').fillna(method='bfill')
            
        # Make sure to remove both date and close from features
        features_df = data.drop([date_col, close_col], axis=1)
        
        # Verify that close is not in features_df (case-insensitive)
        close_cols = [col for col in features_df.columns if col.lower() == 'close']
        if close_cols:
            print("ERROR: 'close' column still found in features after removal!")
            features_df = features_df.drop(close_cols, axis=1)
            
        # Check if features_df is empty
        if features_df.empty:
            raise ValueError("Features DataFrame is empty after removing date and close columns")
            
        # Preprocess only the features (NOT the target)
        print("Preprocessing features (without 'close')")
        features_df, scaler = preprocess_data(features_df)
        
        # Check if preprocessing returned empty DataFrame
        if features_df.empty:
            raise ValueError("Preprocessing resulted in empty DataFrame")
            
        # Keep processed features for LSTM
        features_for_lstm = features_df.copy()
        
        # Create LSTM tensors
        lstm_input = create_lstm_tensors(features_for_lstm, WINDOW_SIZE)
        print("LSTM input shape:", lstm_input.shape)
        
        # Check if we have enough data after creating tensors
        if len(lstm_input) < 10:  # Need at least some data for training
            raise ValueError(f"Insufficient data after creating LSTM tensors: only {len(lstm_input)} samples")
        
        # Process the target variable (y) separately
        print("Processing target variable separately from features")
        y = target.values
        # Trim dates to match the LSTM input length (account for window size)
        dates_trimmed = dates[WINDOW_SIZE-1:]  # This will match the length of lstm_input
        
        # Create a separate scaler for the target variable
        target_scaler = MinMaxScaler(feature_range=(-1, 1))
        y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
        # Trim y_scaled to match LSTM input length
        y_scaled = y_scaled[WINDOW_SIZE-1:]
        
        print(f"LSTM input length: {len(lstm_input)}")
        print(f"Target variable length: {len(y_scaled)}")
        print(f"Dates length: {len(dates_trimmed)}")
        
        # Verify all lengths match
        if not (len(lstm_input) == len(y_scaled) == len(dates_trimmed)):
            raise ValueError(f"Length mismatch: LSTM input ({len(lstm_input)}), target ({len(y_scaled)}), dates ({len(dates_trimmed)})")
        
        # Split data while keeping track of dates
        X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
            lstm_input, y_scaled, dates_trimmed, test_size=0.2, random_state=42
        )
        
        # Setup model and training components
        model = build_lstm_model(X_train)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train).to(device)
        y_train = torch.FloatTensor(y_train).to(device)
        X_test = torch.FloatTensor(X_test).to(device)
        y_test = torch.FloatTensor(y_test).to(device)
        
        # Training loop without early stopping
        epochs = 50
        batch_size = 32
        history = {'loss': [], 'val_loss': []}
        
        print(f"Starting training for {coin_id}...")
        for epoch in range(epochs):
            model.train()
            train_losses = []
            
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
            
            # Calculate losses
            avg_train_loss = np.mean(train_losses)
            history['loss'].append(avg_train_loss)
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_test)
                val_loss = criterion(val_outputs.squeeze(), y_test)
                history['val_loss'].append(val_loss.item())
                
                # Save model with all necessary components
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': val_loss.item(),
                    'input_size': X_train.shape[2],
                    'target_scaler': target_scaler,
                    'feature_scaler': scaler,
                    'window_size': WINDOW_SIZE,
                    'feature_names': features_for_lstm.columns.tolist(),
                    'data_source': 'coingecko' if USE_COINGECKO_ONLY else 'mixed'
                }, f'Stored_data/{coin_id}_lstm_model.pt')
                
                # Print debug information about scalers
                print(f"Feature scaler feature_range: {scaler.feature_range}")
                print(f"Target scaler feature_range: {target_scaler.feature_range}")
                print(f"Number of features saved: {len(features_for_lstm.columns)}")
            
            print(f'Epoch [{epoch+1}/{epochs}] - Training Loss: {avg_train_loss:.4f} - Validation Loss: {val_loss.item():.4f}')
        
        # Evaluation on test set
        model.eval()
        with torch.no_grad():
            # Get predictions and move to CPU before converting to numpy
            test_predictions = model(X_test).squeeze().cpu().numpy()
            y_test_cpu = y_test.cpu().numpy()
            
            # Inverse transform both predictions and actual values to original scale
            test_predictions = target_scaler.inverse_transform(test_predictions.reshape(-1, 1)).flatten()
            y_test_actual = target_scaler.inverse_transform(y_test_cpu.reshape(-1, 1)).flatten()
            
            # Save actual vs predicted prices to Excel
            comparison_df = pd.DataFrame({
                'Date': dates_test,
                'Actual_Price': y_test_actual,
                'Predicted_Price': test_predictions,
                'Absolute_Error': np.abs(y_test_actual - test_predictions),
                'Percentage_Error': np.abs((y_test_actual - test_predictions) / y_test_actual) * 100
            })
            comparison_df.to_excel(f'Stored_data/{coin_id}_actual_vs_predicted.xlsx', index=False)
            
            # Calculate metrics on the original price scale
            mae = np.mean(np.abs(test_predictions - y_test_actual))
            rmse = np.sqrt(np.mean((test_predictions - y_test_actual) ** 2))
            mape = np.mean(np.abs((y_test_actual - test_predictions) / y_test_actual)) * 100

            print(f"Test predictions: {test_predictions[:10]}")
            print(f"Test actual: {y_test_actual[:10]}")
            
            print(f"\nResults for {coin_id}:")
            print(f"Mean Absolute Error: ${mae:.2f}")
            print(f"Root Mean Squared Error: ${rmse:.2f}")
            print(f"Mean Absolute Percentage Error: {mape:.2f}%")
            print(f"Data source: {'CoinGecko only' if USE_COINGECKO_ONLY else 'Mixed sources'}")

            # Create visualization
            plt.figure(figsize=(15, 6))
            
            # Plot 1: Training and Validation Loss
            plt.subplot(1, 2, 1)
            plt.plot(history['loss'], label='Training Loss')
            plt.plot(history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Plot 2: Actual vs Predicted Values
            plt.subplot(1, 2, 2)
            plt.scatter(y_test_actual, test_predictions, alpha=0.5)
            plt.plot([y_test_actual.min(), y_test_actual.max()], 
                     [y_test_actual.min(), y_test_actual.max()], 
                     'r--', lw=2)
            plt.title(f'Actual vs Predicted Values for {coin_id}')
            plt.xlabel('Actual Price ($)')
            plt.ylabel('Predicted Price ($)')
            
            # Add price range information to the plot
            plt.text(0.05, 0.95, f'Price Range: ${y_test_actual.min():.2f} - ${y_test_actual.max():.2f}',
                     transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(f'Stored_data/{coin_id}_training_results.png')
            plt.show()

        return model, scaler, target_scaler
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# Example usage in the main block
if __name__ == "__main__":
    # Train for a specific cryptocurrency
    coin_id = "ket"  # or any other symbol
    model, scaler, target_scaler = train_crypto_model(coin_id)