import pandas as pd
import numpy as np
import torch
from datetime import datetime, timedelta
from DataPreprocessing.data_preprocessing import preprocess_data
from model_architecture.model_architecture import create_lstm_tensors, build_lstm_model

def generate_predictions(num_days=30):
    try:
        # Load data from local file
        data = pd.read_excel("Stored_data/cleaned_data.xlsx")
        print("Loaded data from local file")
        
        # Preprocess the data
        data, scaler = preprocess_data(data)
        
        # Extract close prices and features
        close_prices = data['Close'].values
        features_for_lstm = data.drop('Close', axis=1)
        
        # Create LSTM tensors
        window_size = 35  # Make sure this matches your training window size
        lstm_input = create_lstm_tensors(features_for_lstm, window_size)
        
        # Build model and load weights from local file
        model = build_lstm_model(lstm_input)
        try:
            model.load_state_dict(torch.load('Stored_data/lstm_model.pt'))
            print("Loaded model from local storage")
        except Exception as e:
            print(f"Error loading model: {e}")
            return None, None
        
        model.eval()
        
        # Generate future dates
        last_date = data.index[-1]
        future_dates = [last_date + timedelta(days=x+1) for x in range(num_days)]
        
        # Make predictions
        predictions = []
        current_window = lstm_input[-1:]  # Start with the last window from our data
        
        with torch.no_grad():
            for _ in range(num_days):
                current_window_tensor = torch.FloatTensor(current_window)
                pred = model(current_window_tensor).item()
                predictions.append(pred)
                
                # Update the window for the next prediction
                new_window = current_window[0, 1:, :]
                new_window = np.vstack([new_window, [[pred] * lstm_input.shape[2]]])
                current_window = np.array([new_window])
        
        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted Close': predictions
        })
        
        # Save predictions locally
        predictions_df.to_excel("Stored_data/predictions.xlsx", index=False)
        print("Predictions saved locally")
        
        return predictions_df, scaler
        
    except Exception as e:
        print(f"Error generating predictions: {e}")
        return None, None

def plot_predictions(predictions_df, data):
    # Plot using Plotly Candlestick chart
    fig = go.Figure()
    
    # Add historical candlestick data (OHLC)
    fig.add_trace(go.Candlestick(
        x=data['Date'],
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Historical Prices'
    ))
    
    # Add predicted candlestick data (OHLC)
    fig.add_trace(go.Candlestick(
        x=predictions_df['Date'],
        close=predictions_df['Predicted Close'],
        name='Predicted Price',
        increasing_line_color='orange',
        decreasing_line_color='red',
        opacity=0.6  # Make predicted candles slightly transparent
    ))
    
    # Customize layout for better scrollability and interaction
    fig.update_layout(
        title='Bitcoin Price Prediction (Candlestick)',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        xaxis_rangeslider_visible=True,  # Enable the range slider
        xaxis_type='date',
        hovermode="x unified",
        template='plotly_dark'  # Optional: Dark theme
    )
    
    # Show the plot
    fig.show()

# If you want to run this script standalone for testing:
if __name__ == "__main__":
    predictions_df, data = generate_predictions()
    #plot_predictions(predictions_df, data)
