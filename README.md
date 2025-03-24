# Cryptocurrency Price Prediction API

A FastAPI-based service that provides endpoints for training and predicting cryptocurrency price trends using deep learning models with LSTM architecture.

## Endpoints

### Train Model
**Route:** `POST /train`  
**Request Body:**  
```json
{
  "coin_id": "bitcoin",
  "use_coingecko_only": true
}
```
**Description:** Trains the model for the specified cryptocurrency.  
**Response:** Returns training status and paths to saved model artifacts including:
- Model file
- Predictions file
- Training plots
- Data source information

### Predict Prices
**Route:** `POST /predict`  
**Request Body:**  
```json
{
  "coin_id": "bitcoin",
  "use_coingecko_only": true
}
```
**Description:** Predicts price trends for the specified cryptocurrency using existing data.

### New Prediction
**Route:** `POST /predict_new`  
**Request Body:**  
```json
{
  "coin_id": "bitcoin",
  "use_coingecko_only": true
}
```
**Description:** Fetches fresh data and generates new predictions for the specified cryptocurrency.

### Root
**Route:** `GET /`  
**Description:** Health check endpoint to verify API status.

## Parameters

- `coin_id`: The identifier of the cryptocurrency (e.g., "bitcoin", "ethereum")
- `use_coingecko_only`: Optional boolean parameter to specify whether to use only CoinGecko data (default: true)

## Usage

1. First train a model for your desired cryptocurrency using the `/train` endpoint
2. Use either `/predict` or `/predict_new` to get price predictions
3. The API will automatically handle data fetching, preprocessing, and model management

## Data Sources

The API can use either:
- CoinGecko data only (when `use_coingecko_only` is true)
- Mixed data sources from multiple providers (when `use_coingecko_only` is false)

## Error Handling

The API includes robust error handling for:
- Missing trained models
- Data fetching issues
- Invalid cryptocurrency IDs
- Data preprocessing errors

All errors return appropriate HTTP status codes and descriptive messages.
