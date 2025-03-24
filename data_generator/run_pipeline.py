import logging
from datetime import datetime
import os
from typing import List, Dict, Optional
import pandas as pd
import concurrent.futures
import sys

# Add parent directory to Python path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_generator.crypto_pipeline import get_combined_data

class CryptoDataPipeline:
    """Main driver class for cryptocurrency data collection pipeline."""
    
    def __init__(self, symbols: List[str] = None, start_date: str = "2017-08-17"):
        """
        Initialize the pipeline.
        
        Args:
            symbols: List of cryptocurrency symbols (default: ['BTCUSDT'])
            start_date: Default start date for data collection. For each coin, data will be collected
                       from either this date or its listing date, whichever is later.
        """
        self.symbols = symbols or ['BTCUSDT']
        self.start_date = start_date
        self.results: Dict[str, Optional[pd.DataFrame]] = {}
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging for the pipeline."""
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        
        log_file = os.path.join(
            log_dir, 
            f'crypto_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        )
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
    
    def process_symbol(self, symbol: str) -> Optional[pd.DataFrame]:
        """Process a single cryptocurrency symbol."""
        try:
            logging.info(f"\nProcessing {symbol}...")
            data = get_combined_data(symbol, start_date=self.start_date)
            
            if data is not None:
                # Ensure required columns exist
                date_col = next((col for col in data.columns if col.lower() == 'date'), None)
                close_col = next((col for col in data.columns if col.lower() == 'close'), None)
                
                if not date_col or not close_col:
                    logging.error(f"Required columns 'Date' and/or 'close' not found in data for {symbol}")
                    return None
                
                # Standardize column names
                if date_col != 'Date':
                    data = data.rename(columns={date_col: 'Date'})
                if close_col != 'close':
                    data = data.rename(columns={close_col: 'close'})
                
                # Ensure Date is in datetime format
                data['Date'] = pd.to_datetime(data['Date'])
                
                logging.info(f"Successfully processed {symbol}")
                self._log_data_stats(symbol, data)
                return data
            else:
                logging.error(f"Failed to process {symbol}")
                return None
            
        except Exception as e:
            logging.error(f"Error processing {symbol}: {str(e)}")
            return None
    
    def run_pipeline(self, parallel: bool = True) -> Dict[str, Optional[pd.DataFrame]]:
        """
        Run the data collection pipeline.
        
        Args:
            parallel: Whether to process symbols in parallel
            
        Returns:
            Dictionary mapping symbols to their collected data
        """
        logging.info(f"\nStarting pipeline for {len(self.symbols)} symbols")
        logging.info(f"Symbols to process: {', '.join(self.symbols)}")
        logging.info(f"Default start date: {self.start_date}")
        logging.info("Note: Each symbol will use either the default start date or its listing date, whichever is later.")
        
        if parallel and len(self.symbols) > 1:
            # Process symbols in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_symbol = {
                    executor.submit(self.process_symbol, symbol): symbol 
                    for symbol in self.symbols
                }
                
                for future in concurrent.futures.as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        self.results[symbol] = future.result()
                    except Exception as e:
                        logging.error(f"Error processing {symbol}: {str(e)}")
                        self.results[symbol] = None
        else:
            # Process symbols sequentially
            for symbol in self.symbols:
                self.results[symbol] = self.process_symbol(symbol)
        
        self._log_summary()
        return self.results
    
    def _log_data_stats(self, symbol: str, data: pd.DataFrame):
        """Log statistics about the collected data."""
        logging.info(f"\nStatistics for {symbol}:")
        logging.info(f"- Shape: {data.shape}")
        
        # Check if data has a DatetimeIndex or a Date column
        if isinstance(data.index, pd.DatetimeIndex):
            date_range = pd.date_range(start=data.index.min(), end=data.index.max())
            logging.info(f"- Date Range: {data.index.min()} to {data.index.max()}")
        else:
            date_col = next((col for col in data.columns if col.lower() == 'date'), None)
            if date_col:
                date_range = pd.date_range(start=data[date_col].min(), end=data[date_col].max())
                logging.info(f"- Date Range: {data[date_col].min()} to {data[date_col].max()}")
            else:
                logging.warning("No date column found in data")
                date_range = pd.DatetimeIndex([])
        
        logging.info(f"- Total Trading Days: {len(data)}")
        logging.info(f"- Columns: {', '.join(data.columns)}")
        
        # Calculate data completeness
        if len(date_range) > 0:
            completeness = (len(data) / len(date_range)) * 100
            logging.info(f"- Data Completeness: {completeness:.2f}%")
    
    def _log_summary(self):
        """Log summary of pipeline execution."""
        success_count = sum(1 for df in self.results.values() if df is not None)
        failed_symbols = [symbol for symbol, df in self.results.items() if df is None]
        
        logging.info("\nPipeline Execution Summary:")
        logging.info(f"Total Symbols: {len(self.symbols)}")
        logging.info(f"Successfully Processed: {success_count}")
        logging.info(f"Failed: {len(self.symbols) - success_count}")
        
        if failed_symbols:
            logging.info(f"Failed Symbols: {', '.join(failed_symbols)}")
            
        # Log date ranges for successfully processed symbols
        logging.info("\nDate Ranges for Successfully Processed Symbols:")
        for symbol, df in self.results.items():
            if df is not None:
                if isinstance(df.index, pd.DatetimeIndex):
                    logging.info(f"{symbol}: {df.index.min()} to {df.index.max()} ({len(df)} trading days)")
                else:
                    date_col = next((col for col in df.columns if col.lower() == 'date'), None)
                    if date_col:
                        logging.info(f"{symbol}: {df[date_col].min()} to {df[date_col].max()} ({len(df)} trading days)")
                    else:
                        logging.info(f"{symbol}: {len(df)} trading days (no date column found)")

def run_data_collection(
    symbols: List[str] = None,
    start_date: str = "2017-08-17",
    parallel: bool = True
) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Convenience function to run the data collection pipeline.
    
    Args:
        symbols: List of cryptocurrency symbols
        start_date: Default start date for data collection. For each coin, data will be collected
                   from either this date or its listing date, whichever is later.
        parallel: Whether to process symbols in parallel
        
    Returns:
        Dictionary mapping symbols to their collected data
    """
    pipeline = CryptoDataPipeline(symbols, start_date)
    return pipeline.run_pipeline(parallel)

if __name__ == "__main__":
    # Example usage
    symbols = [
        'BTCUSDT' # Bitcoin
    ]
    
    # Run pipeline with parallel processing
    results = run_data_collection(symbols)
    
    # Access results
    for symbol, data in results.items():
        if data is not None:
            print(f"\n{symbol} data shape: {data.shape}") 