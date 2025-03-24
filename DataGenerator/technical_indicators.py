"""
Technical indicators data fetching module using Twelve Data API - RSI Only Version.
"""

import requests
import pandas as pd
from datetime import datetime
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global constants
BASE_URL = "https://api.twelvedata.com"
DATA_DIR = "Stored_data"
TWELVE_DATA_API_KEY = os.getenv("TWELVE_DATA_API_KEY")
START_DATE = "2017-08-17"

def ensure_data_directory() -> None:
    """Ensure data directory exists"""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        logger.info(f"Created directory: {DATA_DIR}")

def make_api_request(endpoint: str, params: dict) -> dict:
    """Make API request with error handling
    
    Args:
        endpoint: API endpoint to call
        params: Query parameters for the request
        
    Returns:
        JSON response as dictionary or None if request failed
    """
    try:
        params["apikey"] = TWELVE_DATA_API_KEY
        response = requests.get(f"{BASE_URL}/{endpoint}", params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed: {e}")
        return None

def fetch_rsi(symbol: str, time_period: int = 14, interval: str = "1day") -> pd.DataFrame:
    """Fetch RSI data for a symbol
    
    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
        time_period: RSI period (default: 14)
        interval: Time interval between data points (default: "1day")
        
    Returns:
        DataFrame containing RSI data or None if request failed
    """
    ensure_data_directory()
    
    # Calculate days from START_DATE to now
    start = pd.Timestamp(START_DATE)
    end = pd.Timestamp.now()
    outputsize = (end - start).days
    
    # Remove USDT from symbol for Twelve Data API
    clean_symbol = symbol.replace("USDT", "/USD")
    
    logger.info(f"Fetching RSI for {symbol} from {start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}")
    
    api_params = {
        "symbol": clean_symbol,
        "interval": interval,
        "time_period": time_period,
        "outputsize": outputsize,
        "format": "JSON"
    }
    
    data = make_api_request("rsi", api_params)
    
    if not data or "values" not in data:
        logger.error(f"Error fetching RSI: {data.get('message', 'Unknown error') if data else 'No data returned'}")
        return None
    
    # Create DataFrame from response
    df = pd.DataFrame(data["values"])
    
    # Convert datetime and set as index
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.rename(columns={"datetime": "Date"}, inplace=True)
        df.set_index("Date", inplace=True)
    
    # Rename RSI column
    df.rename(columns={"rsi": "rsi14_rsi"}, inplace=True)
    
    # Reset index to have Date as a column
    df.reset_index(inplace=True)
    
    # Log data summary
    logger.info(f"Retrieved {len(df)} RSI data points")
    logger.info(f"Data range: from {df['Date'].min()} to {df['Date'].max()}")
    
    return df

if __name__ == "__main__":
    # Example usage
    symbol = "BTCUSDT"
    rsi_data = fetch_rsi(symbol)
    if rsi_data is not None:
        print("\nRSI Data Sample:")
        print(rsi_data.head())