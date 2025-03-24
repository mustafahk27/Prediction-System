import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.nn.functional as F

def create_lstm_tensors(data, window_size=35):
    """Convert the input DataFrame into tensors for LSTM."""
    # Make a copy to avoid modifying the original
    data = data.copy()
    
    # Check for and remove any non-numeric columns (like Date)
    non_numeric_cols = data.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric_cols:
        print(f"Removing non-numeric columns before tensor creation: {non_numeric_cols}")
        data = data.drop(columns=non_numeric_cols)
    
    # Verify no NaN values
    if data.isnull().sum().sum() > 0:
        print(f"WARNING: Found {data.isnull().sum().sum()} NaN values in data. Filling with 0.")
        data = data.fillna(0)
    
    # Convert to numpy array
    data_values = data.values
    
    # Ensure all values are numeric
    try:
        data_values = data_values.astype(np.float32)
    except ValueError as e:
        print(f"Error converting to float: {e}")
        # Try to identify problematic columns
        for col in data.columns:
            try:
                data[col].astype(np.float32)
            except:
                print(f"Column {col} contains non-numeric values: {data[col].dtype}")
                # Try to convert if possible
                if pd.api.types.is_datetime64_any_dtype(data[col]):
                    print(f"Converting datetime column {col} to numeric (timestamp)")
                    data[col] = data[col].astype(np.int64) // 10**9  # Convert to Unix timestamp
        
        # Try again with cleaned data
        data_values = data.values.astype(np.float32)
    
    # Create the sliding window tensors
    num_samples = data_values.shape[0] - window_size + 1
    X = np.zeros((num_samples, window_size, data_values.shape[1]), dtype=np.float32)
    
    for i in range(num_samples):
        X[i] = data_values[i:i + window_size]
    
    return torch.FloatTensor(X)

class LSTMAttention(nn.Module):
    def __init__(self, input_dim, window_size, hidden_dim=256):
        super(LSTMAttention, self).__init__()
        self.window_size = window_size
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(hidden_dim * window_size, 1)
    
    def forward(self, x):
        lstm_out1, _ = self.lstm1(x)
        lstm_out2, _ = self.lstm2(lstm_out1)
        attention_out, _ = self.attention(lstm_out2, lstm_out2, lstm_out2)
        attention_mul = lstm_out2 * attention_out
        dropout_out = self.dropout(attention_mul)
        flattened = self.flatten(dropout_out)
        output = self.fc(flattened)
        return output

def build_lstm_model(X_train):
    """Build the LSTM model with attention mechanism."""
    if X_train is None:
        raise ValueError("X_train must be provided")
    
    input_dim = X_train.shape[2]
    window_size = X_train.shape[1]
    return LSTMAttention(input_dim, window_size)
