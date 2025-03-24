"""
DataGenerator package for cryptocurrency data collection and preprocessing.
"""

# Make key functions available at package level
from .data_generator import fetch_historical_data, fetch_market_indicators, get_combined_data
from .technical_indicators import fetch_rsi

__all__ = [
    'fetch_historical_data',
    'fetch_market_indicators',
    'fetch_rsi',
    'get_combined_data'
] 