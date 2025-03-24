import os
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time
from binance.client import Client
import pandas_datareader as pdr
import logging
import traceback
import numpy as np
import sys


# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the current directory to the path to find the technical_indicators module
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

load_dotenv()
# Load API key from environment variable
API_KEY = os.getenv('COINGECKO_API_KEY')
if not API_KEY:
    raise ValueError("API key not found. Please set the 'COINGECKO_API_KEY' environment variable.")

# Base URL for CoinGecko API
BASE_URL = "https://pro-api.coingecko.com/api/v3" if API_KEY else "https://api.coingecko.com/api/v3"

# Headers with API key
HEADERS = {"x-cg-pro-api-key": API_KEY} if API_KEY else {}

# Load Binance API keys from environment variables
BINANCE_API_KEY = os.getenv('Binance_api_key')
BINANCE_API_SECRET = os.getenv('Binance_secret_key')

if not BINANCE_API_KEY or not BINANCE_API_SECRET:
    raise ValueError("Binance API key or secret key not found. Please set the 'BINANCE_API_KEY' and 'BINANCE_API_SECRET' environment variables.")

# Try to import technical_indicators using relative import
try:
    from .technical_indicators import fetch_rsi
except ImportError:
    # If that fails, try direct import (when running the file directly)
    try:
        from DataGenerator.technical_indicators import fetch_rsi
        logger.info("Imported technical_indicators as a direct import")
    except ImportError:
        logger.error("Could not import technical_indicators module")
        fetch_rsi = None

# Initialize Binance client
if BINANCE_API_KEY and BINANCE_API_SECRET:
    client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
else:
    logger.warning("Binance API credentials not found. Some functions may not work.")

RSI_API_KEY = os.getenv('RSI_API_KEY')
if not RSI_API_KEY:
    raise ValueError("RSI API key not found. Please set the 'RSI_API_KEY' environment variable.")

# Load LunarCrush API key from environment variable
LUNARCRUSH_API_KEY = os.getenv('LUNARCRUSH_API_KEY')
if not LUNARCRUSH_API_KEY:
    raise ValueError("LunarCrush API key not found. Please set the 'LUNARCRUSH_API_KEY' environment variable.")

# Load FRED API key from environment variable
FRED_API_KEY = os.getenv('FRED_API_KEY')
if not FRED_API_KEY:
    raise ValueError("FRED API key not found. Please set the 'FRED_API_KEY' environment variable.")

TWELVEDATA_API_KEY = os.getenv('TWELVE_DATA_API_KEY')
if not TWELVEDATA_API_KEY:
    raise ValueError("TwelveData API key not found. Please set the 'TWELVEDATA_API_KEY' environment variable.")

NEWS_SENTIMENT_API_KEY = os.getenv('NEWS_SENTIMENT_API_KEY')
if not NEWS_SENTIMENT_API_KEY:
    raise ValueError("News Sentiment API key not found. Please set the 'NEWS_SENTIMENT_API_KEY' environment variable.")

# Import data_preprocessing module
try:
    from DataPreprocessing.data_preprocessing import standardize_date_column
except ModuleNotFoundError:
    # Try alternative import methods
    try:
        # Using relative import (from parent directory)
        sys.path.append(os.path.dirname(parent_dir))
        # Fix the import to not use hyphens in module names
        from DataPreprocessing.data_preprocessing import standardize_date_column
    except (ModuleNotFoundError, ImportError):
        try:
            # Using direct import with modified path
            data_preprocessing_path = os.path.join(parent_dir, 'DataPreprocessing')
            sys.path.append(data_preprocessing_path)
            from DataPreprocessing.data_preprocessing import standardize_date_column
        except (ModuleNotFoundError, ImportError):
            logger.error("Could not import standardize_date_column function")
            # Define a simple version of the function as a fallback
            def standardize_date_column(df):
                df = df.copy()
                date_col = None
                for col in df.columns:
                    if col.lower() in ['date', 'timestamp', 'time']:
                        date_col = col
                        break
                if date_col and date_col != 'Date':
                    df = df.rename(columns={date_col: 'Date'})
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.sort_values('Date')
                    df = df.drop_duplicates(subset=['Date'])
                return df

def fetch_historical_data(coin_id='bitcoin', vs_currency='usd', start_date='2020-01-01', end_date=None):
    """
    Fetch historical market data for a cryptocurrency from CoinGecko.
    :param coin_id: Cryptocurrency ID (e.g., 'bitcoin')
    :param vs_currency: Currency to compare against (e.g., 'usd')
    :param start_date: Start date in 'YYYY-MM-DD' format
    :param end_date: End date in 'YYYY-MM-DD' format
    :return: DataFrame containing historical data
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    # Convert dates to UNIX timestamps
    try:
        start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
    except ValueError:
        print("Invalid date format. Use 'YYYY-MM-DD'.")
        return None

    url = f"{BASE_URL}/coins/{coin_id}/market_chart/range"
    params = {
        "vs_currency": vs_currency,
        "from": start_timestamp,
        "to": end_timestamp
    }

    print(f"Requesting: {url}")
    print(f"Parameters: {params}")

    try:
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()  # Raise error for HTTP issues
        data = response.json()

        # Extract prices, market caps, and total volumes
        prices = pd.DataFrame(data['prices'], columns=['timestamp', 'price'])
        market_caps = pd.DataFrame(data['market_caps'], columns=['timestamp', 'market_cap'])
        total_volumes = pd.DataFrame(data['total_volumes'], columns=['timestamp', 'total_volume'])

        # Convert timestamps to datetime
        prices['timestamp'] = pd.to_datetime(prices['timestamp'], unit='ms')
        market_caps['timestamp'] = pd.to_datetime(market_caps['timestamp'], unit='ms')
        total_volumes['timestamp'] = pd.to_datetime(total_volumes['timestamp'], unit='ms')

        # Merge all data into a single DataFrame
        historical_data = prices.merge(market_caps, on='timestamp').merge(total_volumes, on='timestamp')

        print(f"Successfully fetched {len(historical_data)} records for {coin_id}.")
        return historical_data

    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return None

def fetch_price_marketcap_volume_data(coin_id='bitcoin', vs_currency='usd'):
    """
    Fetch all available historical data for a cryptocurrency in one request.
    """
    # Define the starting point (CoinGecko's earliest available data)
    start_date = "2017-08-17"  

    # Define the current date as the endpoint
    end_date = datetime.now().strftime("%Y-%m-%d")

    # Fetch data for the entire range
    return fetch_historical_data(coin_id, vs_currency, start_date, end_date)

def save_to_csv(data, filename):
    """
    Save DataFrame to a CSV file.
    :param data: DataFrame to save
    :param filename: Name of the CSV file
    """
    try:
        data.to_csv(filename, index=False)
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving file: {e}")

def fetch_binance_ohlc_data(symbol, start_date, end_date=None, interval=Client.KLINE_INTERVAL_1DAY):
    """
    Fetch OHLC data for a cryptocurrency from Binance.
    Includes net order flow calculation to measure buying/selling pressure.
    """
    if client is None:
        logger.error("Binance client not initialized")
        return pd.DataFrame(columns=['Date', 'open', 'high', 'low', 'close', 'volume'])
        
    if end_date is None:
        end_date = datetime.now()
        
    try:
        # Convert dates to strings
        start_str = start_date.strftime('%Y-%m-%d') if isinstance(start_date, datetime) else start_date
        end_str = end_date.strftime('%Y-%m-%d') if isinstance(end_date, datetime) else end_date
        
        # Fetch klines
        klines = client.get_historical_klines(symbol, interval, start_str, end_str)
        
        # Create DataFrame
        ohlc_data = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert timestamps to datetime
        ohlc_data['timestamp'] = pd.to_datetime(ohlc_data['timestamp'], unit='ms')
        
        # Convert price columns to numeric
        for col in ['open', 'high', 'low', 'close', 'volume', 'taker_buy_base_asset_volume']:
            ohlc_data[col] = pd.to_numeric(ohlc_data[col], errors='coerce')
            
        # Calculate net order flow
        ohlc_data['net_order_flow'] = ohlc_data['taker_buy_base_asset_volume'] - (ohlc_data['volume'] - ohlc_data['taker_buy_base_asset_volume'])
        
        # Rename timestamp to Date for consistency
        ohlc_data = ohlc_data.rename(columns={'timestamp': 'Date'})
        
        # Select relevant columns
        ohlc_data = ohlc_data[['Date', 'open', 'high', 'low', 'close', 'volume', 
                              'quote_asset_volume', 'number_of_trades', 
                              'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                              'net_order_flow']]
        
        logger.info(f"Successfully fetched OHLC data with net order flow for {symbol}")
        return ohlc_data
        
    except Exception as e:
        logger.error(f"Error fetching OHLC data for {symbol}: {str(e)}")
        return pd.DataFrame(columns=['Date', 'open', 'high', 'low', 'close', 'volume', 'net_order_flow'])

def fetch_lunarcrush_data(coin_id="bitcoin", start_date="2017-08-17", bucket="day"):
    """
    Fetch social media metrics data from LunarCrush API for a specific cryptocurrency
    
    Parameters:
    - coin_id (str): The cryptocurrency ID to get details for (e.g., 'bitcoin', 'solana')
    - start_date (str): Start date in 'YYYY-MM-DD' format
    - bucket (str): Time aggregation - "day" (default), "hour"
    
    Returns:
    - DataFrame: Processed data in a pandas DataFrame with social metrics
    """
    url = f"https://lunarcrush.com/api4/public/topic/{coin_id}/time-series/v1?bucket=day&interval=all"
    
    # Add bucket parameter if specified
    if bucket:
        url += f"?bucket={bucket}"
    
    headers = {
        "Authorization": f"Bearer {LUNARCRUSH_API_KEY}"
    }
    
    try:
        logger.info(f"Fetching LunarCrush data for {coin_id}...")
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        
        if not data or 'data' not in data:
            logger.warning(f"No LunarCrush data available for {coin_id} or invalid response format")
            return pd.DataFrame(columns=['Date', 'lunar_posts_created', 'lunar_posts_active', 'lunar_contributors_created', 
                                      'lunar_contributors_active', 'lunar_spam', 'lunar_interactions', 'lunar_sentiment'])
        
        # Convert list of dictionaries to DataFrame
        df = pd.DataFrame(data['data'])
        
        # Convert timestamp to datetime
        df['Date'] = pd.to_datetime(df['time'], unit='s')
        
        # Select only the required columns - expanded to include more social metrics
        required_columns = [
            'Date', 'posts_created', 'posts_active', 'contributors_created', 
            'contributors_active', 'spam', 'interactions', 'sentiment',
        ]
        
        # Check if all required columns exist, if not, add them with NaN values
        for col in required_columns:
            if col != 'Date' and col not in df.columns:
                df[col] = float('nan')
        
        # Filter to only include data from the start_date
        df = df[df['Date'] >= pd.to_datetime(start_date)]
        
        # Convert all numeric columns to float
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            df[col] = df[col].astype(float)
        
        # Rename columns to be more descriptive
        column_renames = {
            'posts_created': 'lunar_posts_created',
            'posts_active': 'lunar_posts_active',
            'contributors_created': 'lunar_contributors_created',
            'contributors_active': 'lunar_contributors_active',
            'spam': 'lunar_spam',
            'interactions': 'lunar_interactions',
            'sentiment': 'lunar_sentiment',
        }
        
        # Apply renaming
        for old_col, new_col in column_renames.items():
            if old_col in df.columns:
                df = df.rename(columns={old_col: new_col})
                
        # Update required columns with new names
        required_columns = ['Date'] + [column_renames.get(col, col) for col in required_columns if col != 'Date']
        
        # Select only the required columns for the final DataFrame
        available_columns = [col for col in required_columns if col in df.columns]
        result_df = df[available_columns]
        
        # Check for pre-2020 data
        pre_2020_mask = result_df['Date'] < pd.to_datetime('2020-01-01')
        pre_2020_count = pre_2020_mask.sum()
        
        if pre_2020_count > 0:
            logger.info(f"Found {pre_2020_count} pre-2020 records in LunarCrush data")
            
            # Check for zeros in pre-2020 data
            lunar_columns = [col for col in result_df.columns if col.startswith('lunar_')]
            for col in lunar_columns:
                pre_2020_zeros = (result_df.loc[pre_2020_mask, col] == 0).sum()
                if pre_2020_zeros > 0:
                    logger.warning(f"Column {col}: Found {pre_2020_zeros} zeros in pre-2020 data")
        
        logger.info(f"Successfully fetched LunarCrush data for {coin_id}")
        logger.info(f"Date range: {result_df['Date'].min()} to {result_df['Date'].max()}")
        logger.info(f"Total records: {len(result_df)}")
        logger.info(f"Columns: {result_df.columns.tolist()}")
        
        return result_df
        
    except Exception as e:
        logger.error(f"Error fetching LunarCrush data for {coin_id}: {e}")
        return pd.DataFrame(columns=['Date', 'lunar_posts_created', 'lunar_posts_active', 'lunar_contributors_created', 
                                   'lunar_contributors_active', 'lunar_spam', 'lunar_interactions', 'lunar_sentiment'])

def fetch_all_data(coin_id = 'bitcoin', symbol = 'BTCUSDT', vs_currency = 'usd'):
    """
    Fetch all available data (historical, OHLC, social metrics) for a cryptocurrency.
    :param coin_id: Cryptocurrency ID for CoinGecko (e.g., 'solana')
    :param symbol: Cryptocurrency symbol for Binance (e.g., 'SOLUSDT')
    :param vs_currency: Currency to compare against for CoinGecko (e.g., 'usd')
    :return: Tuple of DataFrames containing historical, OHLC, and social metrics data
    """
    # Define the starting point (CoinGecko's earliest available data)
    start_date = datetime(2013, 4, 28)  # Adjust based on the coin's history if known
    
    # Define the current date as the endpoint
    end_date = datetime.now()
    
    # Fetch historical data from CoinGecko
    historical_data = fetch_historical_data(
        coin_id,
        vs_currency,
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )
    
    # Fetch OHLC data from Binance
    ohlc_data = fetch_binance_ohlc_data(
        symbol,
        start_date,
        end_date
    )
    
    # Fetch social metrics data from LunarCrush
    social_data = fetch_lunarcrush_data(
        coin_id,
        start_date.strftime("%Y-%m-%d")
    )
    
    return historical_data, ohlc_data, social_data

def get_fear_greed_index():
    """
    Get historical and current fear and greed index data.
    :return: DataFrame containing fear and greed index data
    """
    historical_url = 'https://raw.githubusercontent.com/gman4774/Fear_and_Greed_Index/main/all_fng_csv.csv'
    df_historical = pd.read_csv(historical_url)
    df_historical = df_historical.rename(columns={'Fear Greed': 'value'})
    df_historical['Date'] = pd.to_datetime(df_historical['Date'])
    
    start_date = pd.to_datetime('2017-08-17')
    end_date = datetime.now()
    df_historical = df_historical[(df_historical['Date'] >= start_date) & (df_historical['Date'] <= end_date)]

    def fetch_fear_greed_alternative():
        api_url = 'https://api.alternative.me/fng/?limit=1000'
        response = requests.get(api_url)
        data = response.json()
        df = pd.DataFrame(data['data'])
        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
        df.dropna(subset=['timestamp'], inplace=True)
        df['Date'] = pd.to_datetime(df['timestamp'], unit='s')
        df = df.rename(columns={'value': 'value'})
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        return df[['Date', 'value']]

    df_realtime = fetch_fear_greed_alternative()
    df_realtime = df_realtime[df_realtime['Date'] > df_historical['Date'].max()]
    df_merged = pd.concat([df_historical[['Date', 'value']], df_realtime], ignore_index=True)
    full_date_range = pd.date_range(start=start_date, end=end_date)
    df_merged = pd.merge(pd.DataFrame(full_date_range, columns=['Date']), df_merged, on='Date', how='left')
    df_merged['value'] = df_merged['value'].replace(0, pd.NA).ffill()

    def classify_fear_greed(value):
        if value <= 24:
            return 'Extreme Fear'
        elif 25 <= value <= 49:
            return 'Fear'
        elif value == 50:
            return 'Neutral'
        elif 51 <= value <= 74:
            return 'Greed'
        elif value >= 75:
            return 'Extreme Greed'

    df_merged['classification'] = df_merged['value'].apply(classify_fear_greed)
    df_merged.sort_values('Date', inplace=True)
    df_merged = df_merged[df_merged['Date'] <= end_date]
    return df_merged


def calculate_rsi(prices, period=14):
    """
    Calculate RSI for given price data.
    :param prices: Series of prices
    :param period: Period for RSI calculation
    :return: Series containing RSI values
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def fetch_and_calculate_rsi(symbol, start_date="2017-08-17", api_key=RSI_API_KEY):
    """
    Fetch and calculate RSI with pagination support.
    :param symbol: Cryptocurrency symbol (e.g., 'BTCUSDT')
    :param api_key: API key for RSI data
    :param start_date: Start date for data collection
    :return: DataFrame containing RSI data
    """
    try:
        end_date = datetime.now()
        start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        end_timestamp = int(end_date.timestamp())
        
        # Initialize empty DataFrame for all data
        all_data = pd.DataFrame()
        current_timestamp = end_timestamp
        
        while current_timestamp > start_timestamp:
            # Make API request with timestamp
            url = f'https://min-api.cryptocompare.com/data/v2/histoday'
            params = {
                'fsym': symbol[:-4],  # Remove USDT
                'tsym': 'USDT',
                'limit': 2000,
                'toTs': current_timestamp,
                'api_key': api_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if data['Response'] == 'Error':
                print(f"Error fetching RSI data for {symbol}: {data['Message']}")
                return None
            
            # Convert to DataFrame
            df_segment = pd.DataFrame(data['Data']['Data'])
            if df_segment.empty:
                break
                
            # Append to main DataFrame
            all_data = pd.concat([df_segment, all_data], ignore_index=True)
            
            # Update timestamp for next iteration
            current_timestamp = int(df_segment['time'].min()) - 86400  # Subtract one day
            
            # Add small delay to avoid rate limiting
            time.sleep(0.1)
        
        if all_data.empty:
            print(f"No data available for {symbol}")
            return None
            
        # Process the complete dataset
        all_data['Date'] = pd.to_datetime(all_data['time'], unit='s')
        all_data = all_data.sort_values('Date')
        all_data = all_data.drop_duplicates(subset=['Date'])
        
        # Calculate RSI
        all_data['RSI'] = calculate_rsi(all_data['close'])
        
        # Filter date range and clean data
        all_data = all_data[all_data['Date'] >= pd.to_datetime(start_date)]
        all_data = all_data[all_data['RSI'] != 0]
        all_data = all_data.dropna(subset=['RSI'])
        
        if all_data.empty:
            print(f"No valid RSI data available for {symbol}")
            return None
            
        print(f"Successfully calculated RSI for {symbol}")
        print(f"Date range: {all_data['Date'].min()} to {all_data['Date'].max()}")
        print(f"Total records: {len(all_data)}")
        
        # Return only necessary columns without setting index
        return all_data[['Date', 'close', 'RSI']]
        
    except Exception as e:
        print(f"Error in RSI calculation for {symbol}: {str(e)}")
        return None

def get_inflation_data(start_date='2017-08-17'):
    """
    Get inflation data from FRED.
    :param start_date: Start date for data collection
    :return: DataFrame containing inflation data
    """
    end_date = datetime.now()
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    
    cpi_data = pdr.get_data_fred('CPIAUCSL', start_dt, end_date)
    cpi_data['Inflation Rate'] = cpi_data['CPIAUCSL'].pct_change(12) * 100
    
    daily_date_range = pd.date_range(start=start_date, end=end_date.strftime("%Y-%m-%d"))
    daily_inflation_rate = pd.DataFrame(index=daily_date_range, columns=['Inflation Rate'])
    
    annual_inflation_rate = cpi_data['Inflation Rate'].resample('YE').last().dropna()
    annual_inflation_rate.index = annual_inflation_rate.index.year
    
    for year in range(start_dt.year, end_date.year + 1):
        if year in annual_inflation_rate.index:
            mask = (daily_inflation_rate.index.year == year)
            daily_inflation_rate.loc[mask, 'Inflation Rate'] = annual_inflation_rate.loc[year]
    
    daily_inflation_rate.reset_index(inplace=True)
    daily_inflation_rate.rename(columns={'index': 'Date'}, inplace=True)
    return daily_inflation_rate

def get_cpi_data(api_key = NEWS_SENTIMENT_API_KEY, start_date='2017-08-17'):
    """
    Fetch CPI data directly from FRED API.
    :param api_key: FRED API key
    :param start_date: Start date for data collection
    :return: DataFrame containing CPI data
    """
    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        series_id = "CPIAUCSL"  # CPI Series ID for All Urban Consumers (US)
        
        # Ensure start_date is not before the earliest available CPI data
        start_dt = pd.to_datetime(start_date)
        if start_dt.year < 1947:  # FRED CPI data starts from 1947
            start_date = "1947-01-01"
        
        # Construct FRED API URL
        url = (
            f"https://api.stlouisfed.org/fred/series/observations"
            f"?series_id={series_id}"
            f"&observation_start={start_date}"
            f"&observation_end={end_date}"
            f"&api_key={api_key}"
            f"&file_type=json"
        )
        
        # Fetch data from FRED
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Error: FRED API returned status code {response.status_code}")
            return pd.DataFrame(columns=['Date', 'CPI'])
            
        data = response.json()
        
        if "observations" in data:
            # Convert API response to DataFrame
            df = pd.DataFrame(data["observations"])
            df['Date'] = pd.to_datetime(df['date'])
            df['CPI'] = pd.to_numeric(df['value'], errors='coerce')
            
            # Drop original columns and any invalid data
            df = df[['Date', 'CPI']].dropna()
            
            # Create a complete date range from start_date to end_date
            date_range = pd.date_range(start=start_date, end=end_date, name='Date')
            daily_df = pd.DataFrame(index=date_range)
            
            # Merge and interpolate
            df.set_index('Date', inplace=True)
            daily_df = daily_df.join(df['CPI'])
            
            # Forward fill first, then interpolate remaining gaps
            daily_df['CPI'] = daily_df['CPI'].ffill()
            daily_df['CPI'] = daily_df['CPI'].interpolate(method='linear')
            
            # Reset index and ensure column names
            daily_df.reset_index(inplace=True)
            daily_df.columns = ['Date', 'CPI']
            
            print(f"Successfully fetched CPI data from FRED")
            print(f"Date range: {daily_df['Date'].min()} to {daily_df['Date'].max()}")
            print(f"Total records: {len(daily_df)}")
            
            # Filter to requested start date
            daily_df = daily_df[daily_df['Date'] >= pd.to_datetime(start_date)]
            return daily_df
            
        else:
            print(f"Error fetching FRED data: {data.get('error_message', 'Unknown error')}")
            return pd.DataFrame(columns=['Date', 'CPI'])
            
    except Exception as e:
        print(f"Error fetching CPI data: {str(e)}")
        return pd.DataFrame(columns=['Date', 'CPI'])

def fetch_market_indicators(symbol, start_date):
    """
    Fetch market indicators (inflation, CPI, fear & greed index)
    """
    try:
        # Convert start_date to datetime if it's a string
        if isinstance(start_date, str):
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        else:
            # If start_date is already a datetime object, use it directly
            start_dt = start_date
            
        end_date = datetime.now()
        
        # Get inflation data
        try:
            inflation_data = pdr.get_data_fred('CPIAUCSL', start_dt, end_date)
            inflation_data['Inflation Rate'] = inflation_data['CPIAUCSL'].pct_change(12) * 100
            inflation_data = inflation_data.reset_index()[['DATE', 'Inflation Rate']]
            inflation_data = inflation_data.rename(columns={'DATE': 'Date'})
            logger.info("Successfully fetched inflation data")
        except Exception as e:
            logger.warning(f"Could not fetch inflation data: {str(e)}")
            inflation_data = pd.DataFrame(columns=['Date', 'Inflation Rate'])
            
        # Get CPI data
        try:
            cpi_data = pdr.get_data_fred('CPIAUCSL', start_dt, end_date)
            cpi_data = cpi_data.reset_index()[['DATE', 'CPIAUCSL']]
            cpi_data = cpi_data.rename(columns={'DATE': 'Date', 'CPIAUCSL': 'CPI'})
            logger.info("Successfully fetched CPI data")
        except Exception as e:
            logger.warning(f"Could not fetch CPI data: {str(e)}")
            cpi_data = pd.DataFrame(columns=['Date', 'CPI'])
            
        # Get Fear and Greed Index
        try:
            fear_greed_data = get_fear_greed_index()
            fear_greed_data = fear_greed_data[['Date', 'value']].rename(columns={'value': 'fear_greed_index'})
            logger.info("Successfully fetched Fear and Greed Index")
        except Exception as e:
            logger.warning(f"Could not fetch Fear and Greed Index: {str(e)}")
            fear_greed_data = pd.DataFrame(columns=['Date', 'fear_greed_index'])
            
        # Combine all market indicators
        market_dfs = [df for df in [inflation_data, cpi_data, fear_greed_data] 
                     if not df.empty and 'Date' in df.columns]
        
        if not market_dfs:
            logger.warning("No market indicators available")
            return pd.DataFrame(columns=['Date'])
            
        # Merge all market indicators
        market_data = market_dfs[0].copy()
        for df in market_dfs[1:]:
            market_data = market_data.merge(df, on='Date', how='outer')
            
        # Sort and fill missing values
        market_data = market_data.sort_values('Date')
        market_data = market_data.fillna(method='ffill').fillna(method='bfill')
        
        # Ensure numeric columns are float type
        numeric_cols = market_data.select_dtypes(include=[np.number]).columns
        market_data[numeric_cols] = market_data[numeric_cols].astype(float)
        
        logger.info("Successfully combined market indicators")
        return market_data
        
    except Exception as e:
        logger.error(f"Error fetching market indicators: {str(e)}")
        return pd.DataFrame(columns=['Date'])

def get_coin_id_from_symbol(symbol):
    """
    Convert trading symbol to CoinGecko coin ID using CoinGecko's API.
    :param symbol: Trading symbol (e.g., 'BTC', 'BTCUSDT', 'ETH', 'ETHUSDT')
    :return: CoinGecko coin ID (e.g., 'bitcoin', 'ethereum')
    """
    # Remove USDT and convert to lowercase
    base_symbol = symbol.replace('USDT', '').lower()
    
    try:
        # First try direct search with the base symbol
        url = f"{BASE_URL}/search"
        params = {"query": base_symbol}
        
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get('coins') and len(data['coins']) > 0:
            # Get the first match
            coin_id = data['coins'][0]['id']
            logger.info(f"Found CoinGecko ID for {symbol}: {coin_id}")
            return coin_id
            
        # If no direct match, try searching with the full symbol
        url = f"{BASE_URL}/search"
        params = {"query": symbol}
        
        response = requests.get(url, headers=HEADERS, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data.get('coins') and len(data['coins']) > 0:
            coin_id = data['coins'][0]['id']
            logger.info(f"Found CoinGecko ID for {symbol}: {coin_id}")
            return coin_id
            
        # If still no match, try to get the coin list and find the best match
        url = f"{BASE_URL}/coins/list"
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        coins_list = response.json()
        
        # Find the best match based on symbol similarity
        best_match = None
        best_score = 0
        
        for coin in coins_list:
            if coin.get('symbol', '').lower() == base_symbol:
                best_match = coin['id']
                break
            # Calculate similarity score
            score = sum(a == b for a, b in zip(coin.get('symbol', '').lower(), base_symbol))
            if score > best_score:
                best_score = score
                best_match = coin['id']
        
        if best_match:
            logger.info(f"Found best matching CoinGecko ID for {symbol}: {best_match}")
            return best_match
            
        logger.warning(f"No CoinGecko ID found for {symbol}, using base symbol: {base_symbol}")
        return base_symbol
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching CoinGecko data: {e}")
        return base_symbol
    except Exception as e:
        logger.error(f"Unexpected error in get_coin_id_from_symbol: {e}")
        return base_symbol

def handle_pre2020_lunarcrush_data(df):
    """
    Handle missing or zero values in LunarCrush data before 2020.
    
    Args:
        df (pd.DataFrame): DataFrame with LunarCrush columns and a Date column
        
    Returns:
        pd.DataFrame: DataFrame with handled pre-2020 LunarCrush data
    """
    if df is None or df.empty or 'Date' not in df.columns:
        return df
    
    # Ensure Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'])
    
    # Get LunarCrush columns
    lunar_columns = [col for col in df.columns if col.startswith('lunar_')]
    if not lunar_columns:
        return df
    
    logger.info(f"Handling pre-2020 LunarCrush data for {len(lunar_columns)} columns")
    
    # Split data into pre-2020 and post-2020
    pre_2020_mask = df['Date'] < pd.to_datetime('2020-01-01')
    post_2020_mask = ~pre_2020_mask
    
    pre_2020_data = df[pre_2020_mask]
    post_2020_data = df[post_2020_mask]
    
    if pre_2020_data.empty or post_2020_data.empty:
        logger.info("No pre-2020 or post-2020 data available for comparison")
        return df
    
    logger.info(f"Found {len(pre_2020_data)} pre-2020 rows and {len(post_2020_data)} post-2020 rows")
    
    # For each LunarCrush column
    for col in lunar_columns:
        # Check for zeros or NaNs in pre-2020 data
        pre_2020_zeros = (pre_2020_data[col] == 0).sum()
        pre_2020_nans = pre_2020_data[col].isna().sum()
        
        if pre_2020_zeros > 0 or pre_2020_nans > 0:
            logger.info(f"Column {col}: Found {pre_2020_zeros} zeros and {pre_2020_nans} NaNs in pre-2020 data")
            
            # Calculate statistics from post-2020 data
            post_2020_median = post_2020_data[col].median()
            
            # Replace zeros in pre-2020 data with post-2020 median
            if pre_2020_zeros > 0 and not pd.isna(post_2020_median) and post_2020_median > 0:
                df.loc[pre_2020_mask & (df[col] == 0), col] = post_2020_median
                logger.info(f"Imputed {pre_2020_zeros} zeros in pre-2020 {col} with median {post_2020_median:.4f}")
            
            # Replace NaNs in pre-2020 data with post-2020 median
            if pre_2020_nans > 0 and not pd.isna(post_2020_median):
                df.loc[pre_2020_mask & df[col].isna(), col] = post_2020_median
                logger.info(f"Imputed {pre_2020_nans} NaNs in pre-2020 {col} with median {post_2020_median:.4f}")
    
    return df

def get_combined_data(symbol, start_date="2017-08-17"):
    """
    Get combined data from all sources including OHLC, CoinGecko, technical indicators, etc.
    
    Args:
        symbol: Cryptocurrency symbol (e.g., 'BTCUSDT')
        start_date: Start date for data collection
        
    Returns:
        DataFrame containing combined data
    """
    logger.info(f"Getting combined data for {symbol}")
    
    # Convert start_date to datetime if needed
    if isinstance(start_date, str):
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    else:
        start_dt = start_date
        
    # Get coin_id for CoinGecko and LunarCrush
    coin_id = get_coin_id_from_symbol(symbol)
    logger.info(f"Using CoinGecko coin ID: {coin_id}")
    
    # Initialize an empty list to store valid DataFrames
    dataframes = []
    
    # 1. Get OHLC data from Binance
    ohlc_data = fetch_binance_ohlc_data(symbol, start_dt)
    if ohlc_data is not None and not ohlc_data.empty:
        ohlc_data = standardize_date_column(ohlc_data)
        dataframes.append(ohlc_data)
    
    # 2. Get market data from CoinGecko
    try:
        coingecko_data = fetch_historical_data(
            coin_id, 
            'usd', 
            start_dt.strftime("%Y-%m-%d"),
            datetime.now().strftime("%Y-%m-%d")
        )
        if coingecko_data is not None and not coingecko_data.empty:
            if 'timestamp' in coingecko_data.columns:
                coingecko_data = coingecko_data.rename(columns={'timestamp': 'Date'})
            coingecko_data = standardize_date_column(coingecko_data)
            dataframes.append(coingecko_data)
            logger.info(f"Added CoinGecko data with {len(coingecko_data)} rows")
    except Exception as e:
        logger.warning(f"Failed to fetch CoinGecko data: {e}")
    
    # 3. Get technical indicators
    if fetch_rsi is not None:
        try:
            tech_data = fetch_rsi(symbol)
            if tech_data is not None and not tech_data.empty:
                # Ensure date column is named 'Date' and in datetime format
                if 'datetime' in tech_data.columns:
                    tech_data = tech_data.rename(columns={'datetime': 'Date'})
                elif 'date' in tech_data.columns and 'Date' not in tech_data.columns:
                    tech_data = tech_data.rename(columns={'date': 'Date'})
                tech_data['Date'] = pd.to_datetime(tech_data['Date'])
                
                # Verify Date column exists before adding
                if 'Date' in tech_data.columns:
                    dataframes.append(tech_data)
                    logger.info(f"Added technical indicators with {len(tech_data)} rows")
                else:
                    logger.warning("Technical indicators missing Date column")
        except Exception as e:
            logger.warning(f"Failed to fetch technical indicators: {e}")
    
    # 4. Get market indicators (inflation, CPI, fear & greed)
    try:
        market_data = fetch_market_indicators(symbol, start_dt)
        if market_data is not None and not market_data.empty:
            # Ensure date column is named 'Date' and in datetime format
            if 'date' in market_data.columns and 'Date' not in market_data.columns:
                market_data = market_data.rename(columns={'date': 'Date'})
            market_data['Date'] = pd.to_datetime(market_data['Date'])
            
            # Verify Date column exists
            if 'Date' in market_data.columns:
                dataframes.append(market_data)
                logger.info(f"Added market indicators with {len(market_data)} rows")
            else:
                logger.warning("Market indicators missing Date column")
    except Exception as e:
        logger.warning(f"Failed to fetch market indicators: {e}")
    
    # 5. Get RSI data
    try:
        rsi_data = fetch_and_calculate_rsi(symbol, start_date=start_dt.strftime("%Y-%m-%d"))
        if rsi_data is not None and not rsi_data.empty:
            # Ensure date column is named 'Date' and in datetime format
            if 'date' in rsi_data.columns and 'Date' not in rsi_data.columns:
                rsi_data = rsi_data.rename(columns={'date': 'Date'})
            rsi_data['Date'] = pd.to_datetime(rsi_data['Date'])
            
            # Verify Date column exists
            if 'Date' in rsi_data.columns:
                dataframes.append(rsi_data)
                logger.info(f"Added RSI data with {len(rsi_data)} rows")
            else:
                logger.warning("RSI data missing Date column")
    except Exception as e:
        logger.warning(f"Failed to fetch RSI data: {e}")
    
    # 6. Get social data from LunarCrush
    try:
        logger.info(f"Fetching LunarCrush social data for {coin_id}...")
        social_data = fetch_lunarcrush_data(
            coin_id, 
            start_dt.strftime("%Y-%m-%d")
        )
        if social_data is not None and not social_data.empty:
            # Ensure date column is named 'Date' and in datetime format
            if 'date' in social_data.columns and 'Date' not in social_data.columns:
                social_data = social_data.rename(columns={'date': 'Date'})
            social_data['Date'] = pd.to_datetime(social_data['Date'])
            
            # Verify Date column exists
            if 'Date' in social_data.columns:
                dataframes.append(social_data)
                logger.info(f"Added LunarCrush social data with {len(social_data)} rows and {len(social_data.columns)} columns")
                logger.info(f"LunarCrush columns: {social_data.columns.tolist()}")
            else:
                logger.warning("LunarCrush social data missing Date column")
    except Exception as e:
        logger.warning(f"Failed to fetch LunarCrush social data: {e}")
        logger.warning(f"Error details: {str(e)}")

    # 7. EXPLICITLY add Fear and Greed Index
    try:
        logger.info("Fetching Fear and Greed Index data")
        fear_greed_data = get_fear_greed_index()
        if fear_greed_data is not None and not fear_greed_data.empty:
            # Standardize date column format
            fear_greed_data['Date'] = pd.to_datetime(fear_greed_data['Date'])
            
            # We only need 'Date', 'value', and 'classification' columns
            fear_greed_data = fear_greed_data[['Date', 'value', 'classification']]
            
            # Rename columns to be more descriptive
            fear_greed_data = fear_greed_data.rename(columns={
                'value': 'fear_greed_value',
                'classification': 'fear_greed_classification'
            })
            
            dataframes.append(fear_greed_data)
            logger.info(f"Added Fear and Greed Index data with {len(fear_greed_data)} rows")
        else:
            logger.warning("Fear and Greed Index data is empty or None")
    except Exception as e:
        logger.warning(f"Failed to fetch Fear and Greed Index data: {str(e)}")
        logger.warning(traceback.format_exc())
    
    # If no valid DataFrames, return None
    if not dataframes:
        logger.warning(f"No data available for {symbol}")
        return None
    
    # Start with first DataFrame
    logger.info(f"Combining {len(dataframes)} data sources")
    result = dataframes[0].copy()
    
    # Merge remaining DataFrames
    for i, df in enumerate(dataframes[1:], 1):
        if 'Date' in df.columns:
            # Perform merge
            result = result.merge(df, on='Date', how='outer')
            logger.info(f"Merged data source {i+1}, result now has {len(result)} rows")
        else:
            logger.warning(f"Skipping data source {i+1}: missing Date column")
    
    # Sort by date and handle missing values
    result = result.sort_values('Date')
    
    # Check if LunarCrush columns exist in the final dataframe
    lunar_columns = [col for col in result.columns if col.startswith('lunar_')]
    if lunar_columns:
        logger.info(f"Final dataframe contains {len(lunar_columns)} LunarCrush columns: {lunar_columns}")
        
        # Handle pre-2020 LunarCrush data
        result = handle_pre2020_lunarcrush_data(result)
    else:
        logger.warning("No LunarCrush columns found in the final dataframe")
    
    # Handle missing values in LunarCrush columns specifically
    for col in lunar_columns:
        missing_count = result[col].isna().sum()
        if missing_count > 0:
            logger.info(f"Filling {missing_count} missing values in {col}")
            # Forward fill first
            result[col] = result[col].ffill()
            # Then backward fill
            result[col] = result[col].bfill()
            # Finally fill any remaining NaNs with 0
            result[col] = result[col].fillna(0)
    
    # Fill missing values for other columns
    result = result.fillna(method='ffill').fillna(method='bfill')
    
    # Format Date column consistently before saving
    result['Date'] = pd.to_datetime(result['Date']).dt.strftime('%Y-%m-%d %H:%M:%S')
    
    # Log the final column list to verify all our data is included
    logger.info(f"Final combined data contains {len(result)} rows with columns: {', '.join(result.columns)}")
    
    # Save to CSV
    output_dir = "Stored_data"
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{symbol}_combined_data.csv"
    result.to_csv(output_file, index=False)
    logger.info(f"Saved combined data to {output_file}")
    
    return result

if __name__ == "__main__":
    # Example usage
    symbol = "BTCUSDT"  # Symbol for Binance
    start_date = "2017-08-17"  # Start date for data collection
    
    print(f"Fetching combined data for {symbol} from {start_date}...")
    combined_data = get_combined_data(symbol, start_date)
    
    if combined_data is not None and not combined_data.empty:
        print(f"Successfully fetched combined data for {symbol}")
        print(f"Data shape: {combined_data.shape}")
        print(f"Date range: {combined_data['Date'].min()} to {combined_data['Date'].max()}")
        print(f"Columns: {combined_data.columns.tolist()}")
        print(f"Sample data:")
        print(combined_data.head())
    else:
        print(f"Failed to fetch combined data for {symbol}")
